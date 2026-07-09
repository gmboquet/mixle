"""Copula dependence candidates for automatic structure detection (helper for `mixle.inference.estimation`).

Split out of `mixle.inference.estimation` because that module is a high-level compute utility that must
never import concrete `mixle.stats.*` distributions (enforced by
`compute_metadata_test.py::test_high_level_compute_utilities_do_not_import_concrete_distributions`) --
the same reason `learn_bayesian_network`/`bayesian_network_bic` live in `mixle.inference.bayesian_network`
rather than in `estimation.py` itself. `estimation.py` imports only :func:`copula_candidates` from here.
"""

from __future__ import annotations

from typing import Any

from numpy.random import RandomState

# pair-copula free-parameter counts, for BIC when a vine is a candidate (independence adds nothing).
_PAIR_COPULA_PARAMS = {"independence": 0, "gaussian": 1, "clayton": 1, "frank": 1, "gumbel": 1, "student_t": 2}


def _vine_param_count(vine: Any) -> int:
    """Total free parameters of a fitted R-vine: sum of its per-edge pair-copula parameters."""
    return sum(_PAIR_COPULA_PARAMS.get(e.copula.family, 1) for tree in getattr(vine, "trees", []) for e in tree)


def copula_candidates(
    rows: Any, composite: Any, comp_params: int, comp_bic: float, n_log: float, max_its: int, rng: RandomState | None
) -> list[tuple[float, Any, str]]:
    """Copula dependence candidates over the composite's per-field marginals, each scored by BIC.

    Reuses the independently-detected marginals (which the composite already fitted, and which expose the CDF a
    copula needs for the probability-integral transform) and tries dependence cores that a linear-Gaussian
    Bayesian network cannot represent:

    * a **Gaussian copula** -- one correlation matrix, ``d(d-1)/2`` params on top of the marginals; the
      elliptical, tail-independent default.
    * a **regular vine** with Dißmann structure + per-edge family selection -- tried only when the Gaussian
      copula already shows dependence pays (so a vine is never fit on independent data), and kept only if its
      per-edge tail-dependence structure beats the Gaussian core by BIC. This is what lets automatic inference
      recognize joint tail dependence (a Clayton-style joint-crash coupling) instead of forcing it elliptical.

    Returns a list of ``(bic, model, description)`` candidates (possibly empty).
    """
    import numpy as np

    from mixle.inference.estimation import optimize
    from mixle.stats.combinator.copula import CopulaDistribution
    from mixle.stats.multivariate.gaussian_copula import GaussianCopulaDistribution
    from mixle.stats.multivariate.rvine_copula import RVineCopulaDistribution

    marginals = list(getattr(composite, "dists", []) or [])
    if len(marginals) < 2 or any(not callable(getattr(m, "cdf", None)) for m in marginals):
        return []  # a copula needs each marginal's CDF for the probability-integral transform
    d = len(marginals)

    def _fit_bic(core: Any, extra_params: int) -> tuple[float, Any]:
        proto = CopulaDistribution(marginals, core)
        fitted = optimize(rows, proto.estimator(), prev_estimate=proto, max_its=max_its, rng=rng, out=None)
        ll = float(np.sum(fitted.seq_log_density(fitted.dist_to_encoder().seq_encode(rows))))
        return -2.0 * ll + (comp_params + extra_params) * n_log, fitted

    g_bic, gauss = _fit_bic(GaussianCopulaDistribution(np.eye(d)), d * (d - 1) // 2)
    out: list[tuple[float, Any, str]] = [(g_bic, gauss, "copula")]

    # only fit a vine when the (cheaper) Gaussian copula already beat independence -- if there is no dependence
    # to model, the more-flexible vine cannot help and would just cost time. Its per-edge tail-dependence
    # structure then earns its keep only if BIC prefers it over the elliptical Gaussian core.
    if g_bic < comp_bic:
        vproto = CopulaDistribution(marginals, RVineCopulaDistribution(d, []))
        vine = optimize(rows, vproto.estimator(), prev_estimate=vproto, max_its=max_its, rng=rng, out=None)
        v_ll = float(np.sum(vine.seq_log_density(vine.dist_to_encoder().seq_encode(rows))))
        v_bic = -2.0 * v_ll + (comp_params + _vine_param_count(vine.copula)) * n_log
        out.append((v_bic, vine, "vine-copula"))
    return out
