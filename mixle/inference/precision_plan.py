"""Automatic precision allocation from both data and computation.

The control loop that makes "preserve accuracy with minimal compute" the *default*: a fit samples its data
and inspects its model, and where the data is well-conditioned and the model's leaves are float32-safe it
runs the reduced-precision fused kernel; otherwise it stays in float64. Accumulation is always float64 (the
fused kernels promote the reduction), so the result never drifts regardless of the compute band -- the only
question this answers is how cheaply each row can be *scored*. Consulted by ``optimize(precision="minimal")``.

This is the data-aware front of the precision spectrum; a per-leaf affine-tracer allocation (different bands
for different leaves of one model) is the next refinement on top of this whole-model decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Families whose fused float32 score stays within the validated band (summed log-likelihood error < ~1e-6
# relative even on the danger-zone params -- see fused_codegen_test.ReducedPrecisionTest). Any other leaf
# keeps float64.
_FP32_SAFE = frozenset(
    {
        "GaussianDistribution",
        "DiagonalGaussianDistribution",
        "MultivariateGaussianDistribution",
        "ExponentialDistribution",
        "PoissonDistribution",
        "BernoulliDistribution",
        "GeometricDistribution",
        "GammaDistribution",
        "LogGaussianDistribution",
        "CategoricalDistribution",
        "IntegerCategoricalDistribution",
        "BinomialDistribution",
        "NegativeBinomialDistribution",
    }
)


@dataclass
class PrecisionPlan:
    """The allocated compute precision and the reason -- the audit trail of the auto-allocation."""

    compute_dtype: Any
    rationale: str

    def reduced(self) -> bool:
        """Return whether the plan uses lower-than-float64 compute precision."""
        return np.dtype(self.compute_dtype) != np.float64


def _leaf_components(model: Any) -> list[Any]:
    """Flatten a model to its leaf component distributions (through mixtures and composites)."""
    t = type(model).__name__
    if t == "MixtureDistribution":
        return [leaf for c in model.components for leaf in _leaf_components(c)]
    if t == "CompositeDistribution":
        return [leaf for d in model.dists for leaf in _leaf_components(d)]
    return [model]


# Public alias: the typed-runtime compiler declares per-node float32 eligibility from the same
# validated family set the runtime planner uses, so the two can never drift apart.
FP32_SAFE_FAMILIES = _FP32_SAFE


def recommend_compute_precision(
    model: Any,
    data: Any,
    target_rel_error: float = 1e-4,
    sample_size: int = 4096,
    min_variance: float = 1e-6,
    max_magnitude: float = 1e6,
) -> PrecisionPlan:
    """Return the minimal SAFE compute precision (float32 or float64) for fitting ``model`` on ``data``.

    float32 is chosen only when ALL hold: (1) the model fuses (has a reduced-precision kernel), (2) every
    leaf family is float32-safe, (3) no leaf is near-degenerate (variance >= ``min_variance``), and (4) the
    data magnitude is bounded (``|x| <= max_magnitude``, so ``(x-mu)**2`` neither overflows nor loses the
    score's relative precision) -- the regime where the fused float32 summed-LL error is verified ``< ~1e-6``.
    Otherwise float64. (Wide dynamic *range* is not a risk: floating point keeps ~7 relative digits at any
    magnitude; only the absolute magnitude and the variance condition the score.)
    """
    if model is None:
        return PrecisionPlan(np.float64, "no model to inspect -> float64")
    try:
        from mixle.stats.compute.fused_codegen import fusible
    except Exception:  # pragma: no cover - numba optional  # noqa: BLE001
        return PrecisionPlan(np.float64, "fused codegen unavailable -> float64")
    if not fusible(model):
        return PrecisionPlan(np.float64, "model has no fused reduced-precision kernel -> float64")

    # look at the COMPUTATION: leaf families + per-leaf conditioning
    for leaf in _leaf_components(model):
        name = type(leaf).__name__
        if name not in _FP32_SAFE:
            return PrecisionPlan(np.float64, "%s is not float32-safe -> float64" % name)
        s2 = getattr(leaf, "sigma2", None)
        if s2 is not None and float(s2) < min_variance:
            return PrecisionPlan(np.float64, "near-degenerate component (var %.1e) -> float64 for accuracy" % float(s2))

    # look at the DATA: magnitude + dynamic range. Stride across the full dataset rather than taking
    # a leading prefix -- naturally-ordered data (sorted, appended-to over time, grouped by source)
    # can concentrate extreme-magnitude values later in the sequence, which a prefix would never see,
    # silently allocating float32 to data that is not actually well-conditioned for it. When ``data``
    # supports random access, stride-index it directly to avoid materializing the whole sequence;
    # otherwise defer to _numeric_data_sample's own internal stride (it must materialize anyway).
    from mixle.engines.precision import _numeric_data_sample

    if hasattr(data, "__getitem__") and hasattr(data, "__len__"):
        n = len(data)
        if n > sample_size:
            step = n / sample_size
            sample = [data[int(i * step)] for i in range(sample_size)]
        else:
            sample = data
    else:
        sample = data
    s = _numeric_data_sample(sample, sample_size)
    if s is None or s.size == 0:
        return PrecisionPlan(np.float64, "non-numeric / empty data -> float64")
    amax = float(np.max(np.abs(s)))
    if amax > max_magnitude:
        return PrecisionPlan(np.float64, "data magnitude %.1e too large for float32 -> float64" % amax)
    return PrecisionPlan(np.float32, "bounded magnitude (|x|<=%.0e) + fp32-safe leaves -> float32" % amax)
