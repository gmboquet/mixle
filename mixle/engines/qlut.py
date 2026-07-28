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

import math
from collections.abc import Callable
from typing import Any

import numpy as np


def _validate_grid(step: Any, lo: Any, hi: Any) -> tuple[float, int, int]:
    """Validate a finite ordered grid and return its normalized step/code endpoints."""
    if isinstance(step, (bool, np.bool_)):
        raise ValueError("step must be a finite positive number")
    try:
        step = float(step)
        lo = float(lo)
        hi = float(hi)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("step/lo/hi must be finite real numbers") from None
    if not np.isfinite(step) or step <= 0:
        raise ValueError(f"step must be finite and positive, got {step}")
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError(f"lo/hi must be finite, got lo={lo}, hi={hi}")
    if hi <= lo:
        raise ValueError(f"need hi > lo, got lo={lo}, hi={hi}")
    lo_code = int(round(lo / step))
    hi_code = int(round(hi / step))
    if hi_code - lo_code < 1:
        raise ValueError(
            f"step={step} is too coarse for span [{lo}, {hi}]: rounds to a single code "
            "(need at least two grid entries)"
        )
    return step, lo_code, hi_code


def _exact_integer_array(values: Any, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype == np.bool_:
        raise ValueError(f"{name} must contain integer codes, not booleans")
    if np.issubdtype(raw.dtype, np.integer):
        if np.issubdtype(raw.dtype, np.unsignedinteger) and raw.size and np.any(raw > np.iinfo(np.int64).max):
            raise ValueError(f"{name} contains a value outside int64 range")
        return raw.astype(np.int64)
    if np.issubdtype(raw.dtype, np.floating):
        if raw.size and (
            not np.all(np.isfinite(raw))
            or not np.array_equal(raw, np.trunc(raw))
            or np.any(raw < np.iinfo(np.int64).min)
            or np.any(raw >= 2**63)
        ):
            raise ValueError(f"{name} must contain exact finite integer codes")
        return raw.astype(np.int64)
    checked = np.empty(raw.size, dtype=np.int64)
    for i, value in enumerate(raw.ravel()):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must contain exact integer codes")
        if not np.iinfo(np.int64).min <= int(value) <= np.iinfo(np.int64).max:
            raise ValueError(f"{name} contains a value outside int64 range")
        checked[i] = int(value)
    return checked.reshape(raw.shape)


class QuantizedFunction:
    """A scalar function tabulated over quantized inputs; ``f(x) ≈ table[round(x/step)]`` with linear tails."""

    def __init__(self, func: Callable[[np.ndarray], np.ndarray], step: float, lo: float, hi: float) -> None:
        if not callable(func):
            raise TypeError("func must be callable")
        step, lo_code, hi_code = _validate_grid(step, lo, hi)
        self.func = func
        self.step = step
        self.lo_code = lo_code
        self.hi_code = hi_code
        self.lo = self.lo_code * self.step
        self.hi = self.hi_code * self.step
        codes = np.arange(self.lo_code, self.hi_code + 1, dtype=np.int64)
        values = np.asarray(func(codes * self.step))
        if values.shape != codes.shape:
            raise ValueError(f"func must return one value per grid code: expected shape {codes.shape}, got {values.shape}")
        table = np.ascontiguousarray(values, dtype=np.float64)
        if not np.isfinite(table).all():
            raise ValueError("func produced non-finite values over the tabulated range")
        table.setflags(write=False)
        self.table = table
        # boundary slopes for linear extrapolation of unbounded functions outside the table
        self.slope_lo = float((self.table[1] - self.table[0]) / self.step)
        self.slope_hi = float((self.table[-1] - self.table[-2]) / self.step)

    def __call__(self, x: Any) -> np.ndarray:
        """Evaluate via integer code gather (with linear tails); no transcendental runs in range."""
        x = np.asarray(x, dtype=np.float64)
        if x.size and not np.all(np.isfinite(x)):
            raise ValueError("lookup inputs must be finite")
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
        codes = _exact_integer_array(code, "code")
        if codes.size and (np.any(codes < self.lo_code) or np.any(codes > self.hi_code)):
            raise ValueError(f"code must be inside [{self.lo_code}, {self.hi_code}]")
        idx = codes - self.lo_code
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


def _validate_bits_span(bits: int, span: float) -> None:
    if isinstance(bits, (bool, np.bool_)) or not isinstance(bits, (int, np.integer)):
        raise ValueError(f"bits must be an exact non-Boolean integer, got {bits!r}")
    if bits < 1 or bits > 24:
        raise ValueError(f"need 1 <= bits <= 24, got {bits}")
    if not np.isfinite(span) or span <= 0:
        raise ValueError(f"span must be finite and positive, got {span}")


def _prepare_scores_weights(scores: Any, weights: Any) -> tuple[np.ndarray, np.ndarray | None]:
    """Validate and mask ``scores``/``weights`` for the quantized-LSE family.

    Shared by :func:`quantized_logsumexp` and :func:`lse_error_bound` so both operate on identical
    data -- a bound computed from a different mask/shape than the value it's certifying would be
    meaningless.
    """
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
        if not np.isfinite(w).all():
            raise ValueError("weights must be finite")
        if (w < 0).any():
            raise ValueError("weights must be nonnegative")
    keep = ~np.isneginf(scores)  # -inf scores are masked slots (softmax semantics); drop them
    scores = scores[keep]
    if w is not None:
        w = w[keep]
    return scores, w


def _lse_histogram(
    scores: np.ndarray, w: np.ndarray | None, bits: int, span: float
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
    """The integer histogram + exp table :func:`quantized_logsumexp` evaluates, plus which entries clipped.

    ``scores``/``w`` must already be validated and masked (see :func:`_prepare_scores_weights`) and
    non-empty. Returns ``(m, hist, table, clipped)`` where ``clipped[i]`` marks scores that landed more
    than ``span`` below the max ``m`` and were elevated to the bottom bin rather than contributing their
    true (much smaller) mass -- the source of :func:`lse_error_bound`'s tail term.
    """
    m = float(scores.max())
    levels = 1 << bits
    delta = span / levels
    raw = np.rint((scores - m) / delta).astype(np.int64) + levels - 1
    clipped = raw < 0
    idx = np.clip(raw, 0, levels - 1)
    weight_scale = float(np.max(w)) if w is not None else 1.0
    normalized_weights = w / weight_scale if w is not None and weight_scale > 0.0 else w
    hist = np.bincount(idx, weights=normalized_weights, minlength=levels)
    table = np.exp((np.arange(levels) - (levels - 1)) * delta)
    return m, hist, table, clipped, weight_scale


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

    Error: rounding moves each score by at most ``delta/2 = span / 2^(bits+1)``, the grid term of
    :func:`lse_error_bound`. Scores more than ``span`` below the max are CLIPPED UP to the bottom bin
    instead of contributing their true (much smaller) mass. For typical, roughly-bounded score
    distributions that clipped mass is negligible, but it is NOT bounded by the grid term alone -- a
    clipped score paired with a large weight can dominate the sum. Call ``lse_error_bound(bits, span,
    scores=scores, weights=weights)`` with the same data to get a bound that accounts for this rather
    than assuming the grid term is the whole story.
    """
    _validate_bits_span(bits, span)
    scores, w = _prepare_scores_weights(scores, weights)
    if scores.size == 0 or (w is not None and not w.any()):
        return float("-inf")
    m, hist, table, _clipped, weight_scale = _lse_histogram(scores, w, bits, span)
    mass = float(hist @ table)
    if not math.isfinite(mass) or mass <= 0.0 or not math.isfinite(weight_scale) or weight_scale <= 0.0:
        raise ArithmeticError("quantized log-sum-exp accumulation produced invalid mass")
    value = float(np.log(mass) + m + np.log(weight_scale))
    if not math.isfinite(value):
        raise ArithmeticError("quantized log-sum-exp result is not finite")
    return value


def lse_error_bound(bits: int, span: float, *, scores: Any = None, weights: Any = None) -> float:
    """Upper bound on ``|quantized_logsumexp(...) - exact LSE|``.

    With no data (``scores=None``), this is JUST the grid term ``span / 2^(bits+1)`` -- the ordinary
    nearest-code rounding error. That alone is a COMPLETE bound only if no score ever falls more than
    ``span`` below the running max. :func:`quantized_logsumexp` instead CLIPS such scores up to the
    bottom bin, and nothing in ``bits``/``span`` limits how much weight a clipped score can carry, so
    the omitted tail is not generally negligible -- concretely, ``scores=[0, -25],
    weights=[1, 1e12], bits=12, span=24`` has actual error ``~0.96`` against a grid-only bound of
    ``~0.0029``, a ~300x understatement.

    Pass the SAME ``scores``/``weights`` given to (or about to be given to) :func:`quantized_logsumexp`
    to get a bound that also accounts for the clipped tail: the grid term plus ``log1p(clipped_mass /
    in_span_mass)``, where ``clipped_mass`` is the (elevated) linear-domain contribution of out-of-span
    scores and ``in_span_mass`` is the linear-domain contribution of everything else. Derivation: the
    true linear sum is at least the true in-span mass (clipped terms only add) and at most the true
    in-span mass plus the FULLY elevated clipped mass (elevation only increases a clipped term's
    contribution, never decreases it); the true in-span mass is within the grid term (in log-space) of
    the computed in-span mass by the ordinary nearest-code argument. Chaining those bounds and
    converting to log space gives the expression above. Verified empirically: it upper-bounds the exact
    error on the adversarial example above (bound ``~3.67`` vs actual ``~0.96``), stays within ~1e-9 of
    the plain grid term when nothing clips (matching this function's pre-existing behavior), and held
    with zero violations across 20000 randomized trials with huge-dynamic-range weights.
    """
    _validate_bits_span(bits, span)
    grid_term = 0.5 * span / (1 << bits)
    if scores is None:
        if weights is not None:
            raise ValueError("weights requires scores")
        return grid_term
    scores, w = _prepare_scores_weights(scores, weights)
    if scores.size == 0 or (w is not None and not w.any()):
        return 0.0  # quantized_logsumexp returns exactly -inf here too: no rounding occurs
    _m, hist, table, clipped, weight_scale = _lse_histogram(scores, w, bits, span)
    total_mass = float(hist @ table)
    clipped_weight = (
        float((w[clipped] / weight_scale).sum()) if w is not None and weight_scale > 0.0 else float(clipped.sum())
    )
    clipped_mass = clipped_weight * float(table[0])
    in_span_mass = max(0.0, total_mass - clipped_mass)
    if in_span_mass > 0.0:
        tail_term = float(np.log1p(clipped_mass / in_span_mass))
    elif clipped_mass > 0.0:
        tail_term = float("inf")  # every surviving score clipped, and the anchor bin carries zero weight
    else:
        tail_term = 0.0
    return grid_term + tail_term


def error_bound(sup_abs_derivative: float, step: float) -> float:
    """The nearest-code lookup error bound ``(step/2) * sup|f'|`` (e.g. sigmoid sup|f'|=0.25)."""
    if not np.isfinite(sup_abs_derivative) or sup_abs_derivative <= 0:
        raise ValueError(f"sup_abs_derivative must be finite and positive, got {sup_abs_derivative}")
    if not np.isfinite(step) or step <= 0:
        raise ValueError(f"step must be finite and positive, got {step}")
    return 0.5 * step * sup_abs_derivative


def table_bytes(step: float, lo: float, hi: float, itemsize: int = 8) -> int:
    """Bytes a table spanning ``[lo, hi]`` at ``step`` occupies (cache-residency check)."""
    _step, lo_code, hi_code = _validate_grid(step, lo, hi)
    if isinstance(itemsize, (bool, np.bool_)) or not isinstance(itemsize, (int, np.integer)) or itemsize <= 0:
        raise ValueError("itemsize must be a positive exact non-Boolean integer")
    return (hi_code - lo_code + 1) * int(itemsize)


def step_for_tolerance(tol: float, sup_abs_derivative: float) -> float:
    """Largest ``step`` with ``error_bound(sup|f'|, step) <= tol`` -- spend the fewest table entries."""
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError(f"tol must be finite and positive, got {tol}")
    if not np.isfinite(sup_abs_derivative) or sup_abs_derivative <= 0:
        raise ValueError(f"sup_abs_derivative must be finite and positive, got {sup_abs_derivative}")
    return 2.0 * tol / sup_abs_derivative
