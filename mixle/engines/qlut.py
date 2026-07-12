"""Quantized function lookup tables -- every scalar nonlinearity becomes ``table[code]``.

The corollary of quantization: once an operand is the integer code ``round(x/step)``, *any* scalar
function ``f`` is a precomputed table indexed by that code -- no transcendental ever runs. This turns the
nonlinear ops of a model (activations: sigmoid/tanh/GELU/SiLU/softplus; the exp/log that convert between
the log (LNS) and linear domains; the Gaussian-log of :mod:`mixle.engines.lns`) into integer gathers.
Combined with integer-add products and the integer ``logsumexp``, a fully-quantized forward pass has
*no* floating-point exp/log at all.

Nearest-code lookup has error ``<= (step/2) * sup|f'|`` on the tabulated range (e.g. sigmoid: ``0.125*step``).
Unbounded functions (GELU, softplus, ReLU) extrapolate linearly beyond the range using the boundary slope,
so the table need only cover the curved region. Measured 1.5-8x faster than the real transcendental.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


class QuantizedFunction:
    """A scalar function tabulated over quantized inputs; ``f(x) ≈ table[round(x/step)]`` with linear tails."""

    def __init__(self, func: Callable[[np.ndarray], np.ndarray], step: float, lo: float, hi: float) -> None:
        if step <= 0 or hi <= lo:
            raise ValueError("need step > 0 and hi > lo")
        self.func = func
        self.step = float(step)
        self.lo_code = int(round(lo / step))
        self.hi_code = int(round(hi / step))
        self.lo = self.lo_code * self.step
        self.hi = self.hi_code * self.step
        codes = np.arange(self.lo_code, self.hi_code + 1, dtype=np.int64)
        self.table = np.ascontiguousarray(func(codes * self.step), dtype=np.float64)
        # boundary slopes for linear extrapolation of unbounded functions outside the table
        self.slope_lo = float((self.table[1] - self.table[0]) / self.step)
        self.slope_hi = float((self.table[-1] - self.table[-2]) / self.step)

    def __call__(self, x: Any) -> np.ndarray:
        """Evaluate via integer code gather (with linear tails); no transcendental runs in range."""
        x = np.asarray(x, dtype=np.float64)
        q = np.rint(x / self.step).astype(np.int64)
        idx = np.clip(q, self.lo_code, self.hi_code) - self.lo_code
        out = self.table[idx]
        below = q < self.lo_code
        above = q > self.hi_code
        if below.any():
            out = np.where(below, self.table[0] + (x - self.lo) * self.slope_lo, out)
        if above.any():
            out = np.where(above, self.table[-1] + (x - self.hi) * self.slope_hi, out)
        return out

    def lookup(self, code: Any) -> np.ndarray:
        """Gather directly from integer codes already in the quantized domain (the pure-integer path)."""
        idx = np.clip(np.asarray(code, dtype=np.int64), self.lo_code, self.hi_code) - self.lo_code
        return self.table[idx]

    def max_abs_error(self, x: Any) -> float:
        """Empirical max absolute error vs the true function over ``x``."""
        x = np.asarray(x, dtype=np.float64)
        return float(np.max(np.abs(self(x) - self.func(x)))) if x.size else 0.0


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


# Activation factories. Bounded functions (sigmoid/tanh) need only their saturating range; unbounded ones
# (gelu/silu/softplus) cover the curved region and extrapolate linearly past it.
_ACTIVATIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
    "tanh": np.tanh,
    "gelu": _gelu,
    "silu": _silu,
    "softplus": _softplus,
    "relu": lambda x: np.maximum(x, 0.0),
}


def quantized_activation(name: str, step: float = 0.01, span: float = 20.0) -> QuantizedFunction:
    """A common NN activation as a quantized LUT (``sigmoid``/``tanh``/``gelu``/``silu``/``softplus``/``relu``)."""
    if name not in _ACTIVATIONS:
        raise ValueError("unknown activation %r; known: %s" % (name, sorted(_ACTIVATIONS)))
    return QuantizedFunction(_ACTIVATIONS[name], step=step, lo=-span, hi=span)


def quantized_exp(log_step: float = 0.01, lo_log: float = -30.0) -> QuantizedFunction:
    """``exp`` from an LNS log-code (units of ``log_step``) back to the linear domain -- a pure table gather.

    Call with integer log-codes via :meth:`QuantizedFunction.lookup`; this is the softmax/attention
    "back to linear" step, with no real ``exp``.
    """
    return QuantizedFunction(np.exp, step=log_step, lo=lo_log, hi=0.0)


def quantized_logsumexp(scores: Any, *, bits: int = 12, span: float = 24.0, weights: Any = None) -> float:
    """Log-sum-exp via an integer histogram over a ``span/2^bits`` grid plus ONE ``2^bits``-entry exp table.

    Shift by the max, round every score to its grid level, count scores per level (an integer histogram
    -- ``np.bincount``), then a single dot with the exp table and one final ``log``: ``2^bits`` exp
    evaluations total (the table -- 12 bits = 32 KB, cache-resident per :func:`table_bytes`) no matter
    how many scores there are, instead of one ``exp`` per score.

    ``weights`` makes the histogram weighted, which is exactly the group-attention cell form
    (``experiments/group_attention/RESULTS.md``): passing per-cell integer counts computes the LSE of the
    token-level attention mass ``log sum_c count_c * exp(s_c)`` without ever expanding cells back to
    tokens. ``-inf`` scores (masked slots) are dropped, matching softmax semantics.

    Error: rounding moves each score by at most ``delta/2 = span / 2^(bits+1)`` (:func:`lse_error_bound`;
    measured on N(0,3) scores: 8-bit ~2e-3 vs bound 4.7e-2, 12-bit ~6e-5 vs bound 2.9e-3). Scores more
    than ``span`` below the max clip to the bottom bin, adding at most ``mass_clipped * exp(delta - span)``
    in the linear domain -- under 4e-11 relative at the default ``span=24`` -- so the practical bound is
    the grid term.
    """
    if bits < 1 or bits > 24:
        raise ValueError(f"need 1 <= bits <= 24, got {bits}")
    if span <= 0:
        raise ValueError(f"span must be positive, got {span}")
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if scores.size == 0:
        raise ValueError("scores must be non-empty")
    if np.isnan(scores).any() or np.isposinf(scores).any():
        raise ValueError("scores must be free of NaN/+inf")
    if weights is None:
        w = None
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.shape != scores.shape:
            raise ValueError(f"weights shape {w.shape} != scores shape {scores.shape}")
        if (w < 0).any():
            raise ValueError("weights must be nonnegative")
    keep = ~np.isneginf(scores)
    scores = scores[keep]
    if w is not None:
        w = w[keep]
    if scores.size == 0 or (w is not None and not w.any()):
        return float("-inf")

    m = float(scores.max())
    levels = 1 << bits
    delta = span / levels
    idx = np.clip(np.rint((scores - m) / delta).astype(np.int64) + levels - 1, 0, levels - 1)
    hist = np.bincount(idx, weights=w, minlength=levels)
    table = np.exp((np.arange(levels) - (levels - 1)) * delta)
    return float(np.log(float(hist @ table)) + m)


def lse_error_bound(bits: int, span: float) -> float:
    """The grid half-step ``span / 2^(bits+1)`` bounding :func:`quantized_logsumexp`'s rounding error."""
    return 0.5 * span / (1 << bits)


def error_bound(sup_abs_derivative: float, step: float) -> float:
    """The nearest-code lookup error bound ``(step/2) * sup|f'|`` (e.g. sigmoid sup|f'|=0.25)."""
    return 0.5 * step * sup_abs_derivative


def table_bytes(step: float, lo: float, hi: float, itemsize: int = 8) -> int:
    """Bytes a table spanning ``[lo, hi]`` at ``step`` occupies (cache-residency check)."""
    return (int(round(hi / step)) - int(round(lo / step)) + 1) * itemsize


def step_for_tolerance(tol: float, sup_abs_derivative: float) -> float:
    """Largest ``step`` with ``error_bound(sup|f'|, step) <= tol`` -- spend the fewest table entries."""
    if tol <= 0:
        raise ValueError("tol must be positive")
    return 2.0 * tol / sup_abs_derivative
