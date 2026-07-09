"""Per-edge premise checks for cross-modal transport.

Every real modality edge should prove that a plain conditional transport is
usable and calibrated on that edge's own data before the edge is trusted in a
belief graph. Calibration is checked against held-out truth for the edge rather
than transferred from unrelated examples.

This module exposes a reusable per-edge check with two decisions:

* premise fails: the edge should not be used for cross-modal purposes;
* premise passes: the plain conditional transport may be composed as-is.

The check focuses on calibration because a genuine edge usually does not have a
closed-form reference posterior. Coverage of the transport's credible intervals
against held-out truth is the available metric for a real edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import binomtest

from mixle.inference import optimize
from mixle.models.mixture_density import NeuralConditionalDensity, build_mdn

ALPHA = 0.10  # 90% nominal credible-interval coverage
COVERAGE_P_FLOOR = 0.01  # p-value floor below which coverage is inconsistent with nominal


def fit_conditional_transport(
    data,
    *,
    x_dim: int,
    y_dim: int,
    k: int = 3,
    max_its: int = 30,
    m_steps: int = 80,
    lr: float = 3e-3,
    seed: int = 0,
    delta: float | None = 1.0e-9,
    reuse_estep_ll: bool = True,
):
    """Fit ``p(cond | target)`` and return a sampler with ``sample_given``.

    Uses :func:`mixle.models.mixture_density.build_mdn` and
    :class:`~mixle.models.mixture_density.NeuralConditionalDensity`, fit through :func:`optimize`.
    Pass ``delta=None, reuse_estep_ll=False`` for an edge whose relationship needs the full iteration
    budget rather than early stopping.
    """
    module = build_mdn(x_dim=x_dim, y_dim=y_dim, k=k, hidden=32, layers=2)
    leaf = NeuralConditionalDensity(module, m_steps=m_steps, lr=lr)
    fitted = optimize(
        data,
        leaf.estimator(),
        max_its=max_its,
        delta=delta,
        reuse_estep_ll=reuse_estep_ll,
        out=None,
        rng=np.random.RandomState(seed),
    )
    return fitted.sampler(seed=seed)


def marginal_coverage(sampler, x_test, y_test, *, n_draws: int = 200):
    """Return per-dimension credible-interval coverage flags."""
    d = x_test.shape[1]
    covered = [[] for _ in range(d)]
    for i in range(len(x_test)):
        # one batched forward pass for all n_draws of THIS point, instead of n_draws individual calls
        y_batch = np.repeat(np.atleast_2d(np.asarray(y_test[i], dtype=float)), n_draws, axis=0)
        draws = np.asarray(sampler.sample_given_batch(y_batch))
        lo = np.quantile(draws, ALPHA / 2, axis=0)
        hi = np.quantile(draws, 1 - ALPHA / 2, axis=0)
        for dim in range(d):
            covered[dim].append(bool(lo[dim] <= x_test[i, dim] <= hi[dim]))
    return covered


def coverage_consistent_with_nominal(covered_flags) -> tuple[float, float]:
    """``(observed_rate, p_value)`` for a two-sided binomial test of coverage against ``1 - ALPHA``."""
    n = len(covered_flags)
    hits = int(sum(covered_flags))
    p = float(binomtest(hits, n, 1.0 - ALPHA).pvalue)
    return hits / n, p


@dataclass
class EdgeTransportVerdict:
    """Premise decision for one real modality edge, computed on that edge."""

    edge_name: str
    usable: bool
    coverage_rates: list[float] = field(default_factory=list)
    p_values: list[float] = field(default_factory=list)
    reason: str = ""


def verify_edge_transport(edge_name: str, sampler, x_test, y_test, *, n_draws: int = 200) -> EdgeTransportVerdict:
    """Check one fitted edge sampler against held-out calibration data."""
    covered = marginal_coverage(sampler, x_test, y_test, n_draws=n_draws)
    rates, p_values = [], []
    for dim_covered in covered:
        rate, p = coverage_consistent_with_nominal(dim_covered)
        rates.append(rate)
        p_values.append(p)
    failed_dims = [i for i, p in enumerate(p_values) if p <= COVERAGE_P_FLOOR]
    usable = not failed_dims
    reason = "" if usable else f"coverage inconsistent with nominal on dim(s): {failed_dims}"
    return EdgeTransportVerdict(
        edge_name=edge_name, usable=usable, coverage_rates=rates, p_values=p_values, reason=reason
    )


__all__ = [
    "ALPHA",
    "COVERAGE_P_FLOOR",
    "EdgeTransportVerdict",
    "coverage_consistent_with_nominal",
    "fit_conditional_transport",
    "marginal_coverage",
    "verify_edge_transport",
]
