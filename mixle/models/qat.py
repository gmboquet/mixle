"""Quantization-aware training (QAT): straight-through fake-quant, composed by wrapping.

Post-training quantization (:func:`mixle.task.quantize.quantize_mlp`) quantizes an already-trained
float model's weights to int4/int8 *after* the fact -- the model never saw its own quantization
error during training, so gradient descent had no chance to route around it. QAT simulates that
error DURING training instead: every forward pass runs the weight through a real
quantize-then-dequantize round trip (the exact int4 math already in :mod:`mixle.task.quantize` --
this module does not reimplement it), so the optimizer sees a loss landscape shaped by the precision
it will actually run at. The backward pass uses the straight-through estimator (Bengio et al.): the
quantize/dequantize round trip is a step function with zero gradient almost everywhere, so STE simply
copies the incoming gradient through as if the round trip were the identity -- the standard trick
that makes QAT trainable at all.

Composition follows this codebase's established pattern (see ``mixle.experimental.program.lora``,
the peft/LoRA-style wrapper): QAT is a module-level wrapper around ``nn.Linear``, not a change to the
training loop. :class:`~mixle.models.grad_leaf.GradLeaf`'s M-step only ever calls
``module.log_density(x)`` and back-propagates through whatever ``module`` is -- it has no idea some of
that module's Linear layers fake-quantize their weights on every forward call, so QAT drops into
``GradLeaf.fit``/``estimate`` (and, when it exists, the F1 distributed trainer and J4 distillation
students that also route through the same ``log_density`` contract) with **no changes to
``grad_leaf.py``**.

    model = build_causal_lm(vocab=..., d_model=..., n_layer=..., n_head=..., block=...)
    apply_qat(model)                         # every nn.Linear now fake-quantizes its weight to int4
    leaf = GradLeaf(SomeWrapperWithLogDensity(model))
    fitted = leaf.estimator().estimate(None, suff_stat)   # trains QAT-aware, unmodified M-step

F1 (a real distributed trainer skeleton) and J4 (distillation students) are separate, not-yet-built
roadmap items; this module does not depend on either. What is real and tested here: the STE fake-quant
op itself, and that wrapping a real transformer's Linear layers with it and training end-to-end beats
post-hoc PTQ at matched int4 size (see ``mixle/tests/qat_test.py``).
"""

from __future__ import annotations

from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is optional
    _HAS_TORCH = False

__all__ = [
    "fake_quantize",
    "fake_quantize_int4",
    "QATWrapper",
    "apply_qat",
    "set_fake_quant_enabled",
]


def _normalize_quant_config(bits: Any, clip_percentile: Any) -> tuple[int, float | None]:
    import numbers

    from mixle.task.quantize import _QMAX

    if isinstance(bits, bool) or not isinstance(bits, numbers.Integral) or int(bits) not in _QMAX:
        raise ValueError(f"bits must be one of {sorted(_QMAX)}, got {bits}")
    if clip_percentile is not None:
        if isinstance(clip_percentile, bool) or not isinstance(clip_percentile, numbers.Real):
            raise ValueError("clip_percentile must be in (0, 100]")
        clip_percentile = float(clip_percentile)
        if not 0.0 < clip_percentile <= 100.0:
            raise ValueError("clip_percentile must be in (0, 100]")
    return int(bits), clip_percentile


if _HAS_TORCH:

    class _FakeQuantSTE(torch.autograd.Function):
        """Straight-through fake-quant: forward is REAL quantize->dequantize (reusing
        :func:`mixle.task.quantize.quantize_dequantize_array`, the same math PTQ uses), backward is
        the identity -- ``d(loss)/d(input)`` passes through unchanged, per the straight-through
        estimator (Bengio, Léonard & Courville, 2013)."""

        @staticmethod
        def forward(ctx: Any, x: Any, bits: int, clip_percentile: float | None) -> Any:
            from mixle.task.quantize import _QMAX

            qmax = _QMAX[bits]
            detached = x.detach()
            absolute = detached.abs()
            if clip_percentile is None:
                magnitude = absolute.max()
            else:
                # torch.quantile does not implement every low-precision CPU dtype. Promotion to fp32
                # is lossless for fp16/bfloat16 and stays on the declared device; float64 remains
                # float64, so QAT never introduces the old unrelated float32 precision bottleneck.
                quantile_input = (
                    absolute
                    if absolute.dtype in (torch.float32, torch.float64)
                    else absolute.to(dtype=torch.float32)
                )
                magnitude = torch.quantile(quantile_input.reshape(-1), clip_percentile / 100.0).to(x.dtype)
            scale = torch.where(magnitude > 0, magnitude / qmax, torch.ones_like(magnitude))
            return torch.clamp(torch.round(detached / scale), -qmax, qmax) * scale

        @staticmethod
        def backward(ctx: Any, grad_output: Any) -> tuple[Any, None, None]:
            return grad_output, None, None  # the STE trick: copy the gradient through unchanged

    class QATWrapper(nn.Module):
        """Wrap an ``nn.Linear`` so its weight is straight-through fake-quantized on every forward
        call: the module computes ``F.linear(x, fake_quantize(weight), bias)`` instead of
        ``F.linear(x, weight, bias)``. Bias stays fp32, matching PTQ's scheme
        (:func:`mixle.task.quantize.quantize_mlp`) where only weights are quantized.

        Drop-in composition: ``QATWrapper(linear)`` has the same ``forward(x) -> Tensor`` contract as
        the ``Linear`` it wraps, so it slots into any module tree (see :func:`apply_qat`) without the
        surrounding model or training loop changing at all.
        """

        def __init__(
            self, base: Any, *, bits: int = 4, clip_percentile: float | None = None, enabled: bool = True
        ) -> None:
            super().__init__()
            if not isinstance(base, nn.Linear):
                raise TypeError(f"QATWrapper wraps an nn.Linear, got {type(base).__name__}")
            self.base = base
            self.configure(bits=bits, clip_percentile=clip_percentile)
            # a plain attribute (not a buffer/parameter): flip off to run this layer at its real fp32
            # weight -- e.g. to check a QAT-trained model's full-precision quality (see
            # set_fake_quant_enabled), with no change to the wrapped Linear or the surrounding model.
            if not isinstance(enabled, bool):
                raise ValueError("enabled must be a boolean")
            self.enabled = enabled

        def configure(self, *, bits: int, clip_percentile: float | None) -> None:
            """Explicitly replace the quantizer configuration without adding another wrapper."""
            self.bits, self.clip_percentile = _normalize_quant_config(bits, clip_percentile)

        @property
        def weight(self) -> Any:
            return self.base.weight

        @property
        def bias(self) -> Any:
            return self.base.bias

        def forward(self, x: Any) -> Any:
            w = (
                fake_quantize(self.base.weight, bits=self.bits, clip_percentile=self.clip_percentile)
                if self.enabled
                else self.base.weight
            )
            return F.linear(x, w, self.base.bias)

        def extra_repr(self) -> str:
            return f"bits={self.bits}, in={self.base.in_features}, out={self.base.out_features}"

else:  # pragma: no cover - torch is optional

    class QATWrapper:  # type: ignore[no-redef]
        """Torch-absent stand-in: keeps ``QATWrapper`` always importable (``mixle.models`` imports it
        unconditionally, matching every other torch-optional model in this package -- see
        :class:`~mixle.models.gaussian_process.GaussianProcessRegressor`), raising only when actually
        constructed, since it must subclass ``nn.Module`` to be real and ``nn`` does not exist here."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("QATWrapper requires torch.")


def fake_quantize(x: Any, *, bits: int = 4, clip_percentile: float | None = None) -> Any:
    """Straight-through fake-quantize ``x`` to ``bits`` (int4 or int8, per
    :data:`mixle.task.quantize._QMAX`): forward returns the real quantize->dequantize round trip,
    backward passes the gradient through unchanged (STE)."""
    if not _HAS_TORCH:  # pragma: no cover - torch is optional
        raise ImportError("mixle.models.qat requires torch")
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch tensor")
    if not x.is_floating_point() or x.is_complex():
        raise ValueError("x must use a real floating-point dtype")
    if x.numel() == 0:
        raise ValueError("x must not be empty")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("x must contain only finite values")
    bits, clip_percentile = _normalize_quant_config(bits, clip_percentile)
    return _FakeQuantSTE.apply(x, bits, clip_percentile)


def fake_quantize_int4(x: Any, *, clip_percentile: float | None = None) -> Any:
    """``fake_quantize(x, bits=4)`` -- the int4 case this roadmap item targets."""
    return fake_quantize(x, bits=4, clip_percentile=clip_percentile)


def apply_qat(
    model: Any,
    *,
    bits: int = 4,
    clip_percentile: float | None = None,
    reconfigure: bool = False,
) -> Any:
    """Replace every ``nn.Linear`` under ``model`` in place with a :class:`QATWrapper`, so the whole
    model trains quantization-aware (straight-through int4 fake-quant on every Linear weight, every
    forward call). Mirrors ``mixle.experimental.program.lora``'s wrapping pattern: walk
    ``named_children``, swap ``Linear`` leaves, recurse into everything else. Weight-tied layers
    (e.g. ``CausalLM.head`` sharing ``tok.weight``) wrap cleanly -- only the wrapped module's
    ``forward`` changes, the underlying ``nn.Parameter`` (and anything else pointing at it) is
    untouched. Returns ``model`` (mutated in place) for chaining.
    """
    if not _HAS_TORCH:  # pragma: no cover - torch is optional
        raise ImportError("mixle.models.qat requires torch")
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(reconfigure, bool):
        raise ValueError("reconfigure must be a boolean")

    # Validate once even when the model has no Linear leaves, without allocating parameters or
    # advancing the caller's random-number generator.
    bits, clip_percentile = _normalize_quant_config(bits, clip_percentile)

    def replace(m: Any) -> None:
        for name, child in list(m.named_children()):
            if isinstance(child, QATWrapper):
                requested = (bits, clip_percentile)
                current = (child.bits, child.clip_percentile)
                if current != requested:
                    if not reconfigure:
                        raise ValueError(
                            "model already contains a QATWrapper with a different configuration; "
                            "pass reconfigure=True to update it explicitly"
                        )
                    child.configure(bits=bits, clip_percentile=clip_percentile)
                continue
            if isinstance(child, nn.Linear):
                setattr(m, name, QATWrapper(child, bits=bits, clip_percentile=clip_percentile))
            else:
                replace(child)

    replace(model)
    return model


def set_fake_quant_enabled(model: Any, enabled: bool) -> Any:
    """Toggle every :class:`QATWrapper` under ``model`` on/off in place. ``enabled=False`` runs the
    model at its real fp32 weights (e.g. to check that QAT training didn't wreck full-precision
    quality); ``enabled=True`` (the default after :func:`apply_qat`) restores the fake-quant forward.
    Returns ``model`` for chaining.
    """
    if _HAS_TORCH:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a boolean")
        for m in model.modules():
            if isinstance(m, QATWrapper):
                m.enabled = enabled
    return model
