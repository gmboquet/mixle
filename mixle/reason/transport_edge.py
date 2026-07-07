"""Per-edge cross-modal transport premise check (CARD F2-a, workstream F2).

TRANSPORT-a (the F0 gate) cleared a plain mixture-density conditional transport on two TOY inverses
before any belief graph was trusted to rest on it -- but that GO decision was scoped to those toys.
Per the plan, EVERY real modality edge must re-prove the same premise on ITS OWN data before an
equivariant (A3 quotient) refinement is even considered: "PREMISE FIRST -- re-prove a plain
conditional transport is usable + calibrated on THAT pair ... THEN the equivariant refinement only
if the plain conditional is usable."

This module promotes TRANSPORT-a's fit + calibration-check machinery (mixle/tests/transport_proof_test.py)
into a reusable per-edge check, with the same two kill criteria:

  * premise fails  -> the edge does not exist for cross-modal purposes; route around it / drop the pair.
  * premise passes -> the plain conditional is kept AS-IS. No equivariant refinement is attempted here:
    A3-a's own research spike already recorded a negative result for the quotient-leaf refinement
    (no measured sample/parameter-efficiency win), so there is nothing to graft on even when the
    premise clears.

Calibration only -- deliberately no closed-form/reference-posterior check, since a genuine edge (unlike
TRANSPORT-a's toys) has none: coverage of the transport's own credible intervals against held-out
truth is the only metric a real edge can offer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import binomtest

from mixle.inference import optimize
from mixle.models.mixture_density import NeuralConditionalDensity, build_mdn

ALPHA = 0.10  # 90% nominal credible-interval coverage
COVERAGE_P_FLOOR = 0.01  # binomial-test p-value floor below which coverage is NOT consistent with nominal


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
    """Fit ``p(cond | target)`` via a mixture density network and return a SAMPLER exposing
    ``sample_given(cond) -> target``; ``data`` is ``(cond, target)`` pairs.

    Same family and defaults TRANSPORT-a validated (:func:`mixle.models.mixture_density.build_mdn` +
    :class:`~mixle.models.mixture_density.NeuralConditionalDensity`, fit through :func:`optimize`).
    Pass ``delta=None, reuse_estep_ll=False`` for an edge whose relationship needs the full iteration
    budget rather than early-stopping (TRANSPORT-a's own nonlinear case needed this).
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
    """Per-dimension credible-interval coverage: for each held-out ``(x, y)``, does the
    ``[alpha/2, 1-alpha/2]`` quantile interval of ``n_draws`` posterior samples (given ``y``) contain
    the true ``x``, per dimension?"""
    d = x_test.shape[1]
    covered = [[] for _ in range(d)]
    for i in range(len(x_test)):
        draws = np.asarray([sampler.sample_given(y_test[i]) for _ in range(n_draws)])
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
    """The premise-first decision for ONE real modality edge -- computed, not assumed to transfer
    from any other edge's (or TRANSPORT-a's toy) verdict."""

    edge_name: str
    usable: bool
    coverage_rates: list[float] = field(default_factory=list)
    p_values: list[float] = field(default_factory=list)
    reason: str = ""


def verify_edge_transport(edge_name: str, sampler, x_test, y_test, *, n_draws: int = 200) -> EdgeTransportVerdict:
    """Run the premise check for one edge: fit (via :func:`fit_conditional_transport`) happens
    upstream; this checks the RESULTING sampler's calibration against held-out truth. ``usable`` is
    True only if every dimension's coverage is statistically consistent with the nominal rate."""
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
