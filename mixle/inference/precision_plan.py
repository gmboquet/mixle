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

import hashlib
from dataclasses import dataclass
from numbers import Integral, Real
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
    """A validated precision recommendation, distinct from its eventual execution receipt."""

    compute_dtype: Any
    rationale: str
    target_rel_error: float | None = None
    observed_rel_error: float | None = None
    validation_count: int = 0
    context_digest: str | None = None
    executed_dtype: Any = None
    execution_status: str = "not_executed"
    fallback: str | None = None

    def reduced(self) -> bool:
        """Return whether the plan uses lower-than-float64 compute precision."""
        return np.dtype(self.compute_dtype) != np.float64

    def record_execution(self, dtype: Any, *, fallback: str | None = None) -> None:
        """Record the dtype that actually entered the execution path."""
        self.executed_dtype = np.dtype(dtype)
        self.execution_status = "executed"
        self.fallback = fallback


def _has_non_finite(value: Any, _depth: int = 0) -> bool:
    """Whether ``value`` contains a NaN or infinity anywhere in a record-shaped structure.

    Reduced precision cannot be *validated* against data the reference path itself cannot score, and
    non-finite input is the common way that happens. Without this the fallback still occurred, but
    the rationale reported whatever incidental exception fired first -- for a Gaussian leaf, a
    support ContractError naming ``x in (-inf,inf)`` -- which tells a caller nothing about the
    actual problem being a NaN in their data.
    """
    if _depth > 8:  # records nest shallowly; bail rather than walk a cycle
        return False
    if isinstance(value, (bool, np.bool_)) or value is None:
        return False
    if isinstance(value, (int, np.integer)):
        return False
    if isinstance(value, (float, np.floating)):
        return not np.isfinite(value)
    if isinstance(value, np.ndarray):
        return value.dtype.kind in "fc" and not bool(np.isfinite(value).all())
    if isinstance(value, dict):
        return any(_has_non_finite(v, _depth + 1) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_non_finite(v, _depth + 1) for v in value)
    return False


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
    if isinstance(target_rel_error, bool) or not isinstance(target_rel_error, Real):
        raise TypeError("target_rel_error must be a finite positive real number")
    target_rel_error = float(target_rel_error)
    if not np.isfinite(target_rel_error) or target_rel_error <= 0.0:
        raise ValueError("target_rel_error must be a finite positive real number")
    if isinstance(sample_size, bool) or not isinstance(sample_size, Integral):
        raise TypeError("sample_size must be a positive integer")
    sample_size = int(sample_size)
    if sample_size < 1:
        raise ValueError("sample_size must be a positive integer")
    for value, name in ((min_variance, "min_variance"), (max_magnitude, "max_magnitude")):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a finite positive real number")
        if not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be a finite positive real number")
    min_variance, max_magnitude = float(min_variance), float(max_magnitude)

    def plan(dtype, rationale, *, observed=None, count=0, digest=None, fallback=None):
        return PrecisionPlan(
            dtype,
            rationale,
            target_rel_error=target_rel_error,
            observed_rel_error=observed,
            validation_count=count,
            context_digest=digest,
            fallback=fallback,
        )

    if model is None:
        return plan(np.float64, "no model to inspect -> float64", fallback="missing_model")
    try:
        from mixle.stats.compute.fused_codegen import fusible
    except Exception:  # pragma: no cover - numba optional  # noqa: BLE001
        return plan(np.float64, "fused codegen unavailable -> float64", fallback="fused_codegen_unavailable")
    # bare_bridge=False: bare-bridge fusion computes its per-component score tables through each
    # factor's native float64 seq_log_density, so a float32 band would touch only the softmax --
    # no real reduced-precision win to recommend there.
    if not fusible(model, bare_bridge=False):
        return plan(
            np.float64,
            "model has no fused reduced-precision kernel -> float64",
            fallback="not_fusible",
        )

    # look at the COMPUTATION: leaf families + per-leaf conditioning
    for leaf in _leaf_components(model):
        name = type(leaf).__name__
        if name not in _FP32_SAFE:
            return plan(np.float64, "%s is not float32-safe -> float64" % name, fallback="unsafe_family")
        s2 = getattr(leaf, "sigma2", None)
        if s2 is not None and float(s2) < min_variance:
            return plan(
                np.float64,
                "near-degenerate component (var %.1e) -> float64 for accuracy" % float(s2),
                fallback="near_degenerate_component",
            )

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
        sample = list(data)
    s = _numeric_data_sample(sample, sample_size)
    if s is None or s.size == 0:
        return plan(np.float64, "non-numeric / empty data -> float64", fallback="unvalidated_data")
    if not np.all(np.isfinite(s)):
        # NaN/Inf anywhere in the sample must route to the safe fallback EXPLICITLY (MXR-080-0145,
        # mirroring mixle.engines.precision.auto_precision's identical guard): IEEE-754 defines every
        # comparison against NaN as False, so a NaN `amax` would make `amax > max_magnitude` below
        # silently evaluate to False and fall through to the OPTIMISTIC float32 branch instead of the
        # safe float64 fallback -- the opposite of what "risk could not be computed" should mean.
        return plan(np.float64, "non-finite data (NaN/Inf) -> float64", fallback="nonfinite_data")
    amax = float(np.max(np.abs(s)))
    if amax > max_magnitude:
        return plan(
            np.float64,
            "data magnitude %.1e too large for float32 -> float64" % amax,
            fallback="magnitude_limit",
        )

    sample_records = list(sample)
    digest_payload = (
        f"{type(model).__module__}.{type(model).__qualname__}|{repr(model)}|{repr(sample_records)}"
    ).encode()
    context_digest = hashlib.sha256(digest_payload).hexdigest()
    if any(_has_non_finite(record) for record in sample_records):
        return plan(
            np.float64,
            "sample data contains non-finite values, so float32 agreement cannot be validated -> float64",
            count=len(sample_records),
            digest=context_digest,
            fallback="non_finite_data",
        )
    try:
        from mixle.stats.compute.fused_codegen import fused_seq_log_density

        encoding = model.dist_to_encoder().seq_encode(sample_records)
        reference = np.asarray(model.seq_log_density(encoding), dtype=np.float64)
        reduced = np.asarray(
            fused_seq_log_density(model, encoding, compute_dtype=np.float32),
            dtype=np.float64,
        )
        if reference.shape != reduced.shape or reference.size == 0:
            raise RuntimeError("precision validation produced empty or mismatched score arrays")
        if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(reduced)):
            raise RuntimeError("precision validation produced non-finite scores")
        observed_error = float(np.max(np.abs(reduced - reference) / np.maximum(np.abs(reference), 1.0)))
    except Exception as error:  # noqa: BLE001 - any failed validation must fail closed to float64
        # Carry the exception's own message, not just its class. Falling back to float64 is right
        # either way, but the rationale is what a caller reads to find out *why* their model cannot
        # run reduced -- and "ContractError" alone does not distinguish non-finite scores from a
        # shape mismatch from an unsupported leaf. The raises above go to the trouble of saying
        # which; discarding that left the plan reporting a fallback nobody could act on.
        return plan(
            np.float64,
            f"float32 validation did not execute successfully ({type(error).__name__}: {error}) -> float64",
            count=len(sample_records),
            digest=context_digest,
            fallback="validation_failed",
        )
    if observed_error > target_rel_error:
        return plan(
            np.float64,
            f"observed float32 relative score error {observed_error:.3e} exceeds "
            f"target {target_rel_error:.3e} -> float64",
            observed=observed_error,
            count=len(sample_records),
            digest=context_digest,
            fallback="relative_error_exceeded",
        )
    return plan(
        np.float32,
        f"observed float32 relative score error {observed_error:.3e} <= target "
        f"{target_rel_error:.3e} on {len(sample_records)} rows",
        observed=observed_error,
        count=len(sample_records),
        digest=context_digest,
    )
