"""Information-theoretic Bayesian optimization: Max-value Entropy Search (MES).

The improvement-based acquisitions (EI/PI/UCB) score a candidate by how much it might beat the current
best. MES (Wang & Jegelka 2017) instead scores a candidate by how much evaluating it would reduce the
*entropy of the global optimum value* ``y* = max f`` -- the mutual information ``I(y; y* | x)``. It is
often more sample-efficient and low-overhead: with a GP, ``I(y; y*|x)`` has a closed form per sampled
``y*`` (a truncated-Gaussian entropy), and plausible ``y*`` are drawn by fitting a Gumbel to the
distribution of the maximum over a candidate set.

``max_value_entropy_search`` is pure NumPy (given posterior moments and ``y*`` samples); the
:func:`propose_mes` driver fits the torch GP surrogate, so it needs PyTorch.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.stats import norm

from mixle.doe.bayesopt import _fit_surrogate, _validate_xy
from mixle.doe.designs import Bounds, _as_bounds, _as_rng, latin_hypercube

# y_r = a - b * log(-log r) for the Gumbel; constants at r = 0.25, 0.5, 0.75.
_C25, _C50, _C75 = float(np.log(-np.log(0.25))), float(np.log(-np.log(0.5))), float(np.log(-np.log(0.75)))


def sample_max_values(mean: Any, std: Any, n_samples: int = 64, *, seed: int | RandomState = 0) -> np.ndarray:
    """Sample plausible global-max values ``y*`` via the Gumbel approximation (Wang & Jegelka 2017).

    The CDF of the maximum over the candidate cloud is ``P(max <= y) = prod_i Phi((y - mu_i)/sd_i)``; a
    Gumbel is fit to its 25/50/75 percentiles (found by bisection) and sampled. Returns an
    ``(n_samples,)`` array of ``y*`` draws (never below the best posterior mean).
    """
    mu = np.asarray(mean, dtype=np.float64).ravel()
    sd = np.maximum(np.asarray(std, dtype=np.float64).ravel(), 1e-9)
    rng = seed if isinstance(seed, RandomState) else RandomState(seed)

    def cdf_max(y: float) -> float:
        return float(np.exp(np.sum(norm.logcdf((y - mu) / sd))))

    lo = float((mu - 5.0 * sd).min())
    hi = float((mu + 8.0 * sd).max())

    def quantile(target: float) -> float:
        a, b = lo, hi
        for _ in range(60):
            m = 0.5 * (a + b)
            if cdf_max(m) < target:
                a = m
            else:
                b = m
        return 0.5 * (a + b)

    y25, y50, y75 = quantile(0.25), quantile(0.5), quantile(0.75)
    scale = max((y75 - y25) / (_C25 - _C75), 1e-6)  # b from the IQR
    loc = y50 - scale * _C50  # a from the median
    u = rng.uniform(1e-6, 1.0 - 1e-6, int(n_samples))
    ystar = loc - scale * np.log(-np.log(u))
    return np.maximum(ystar, mu.max())


def max_value_entropy_search(mean: Any, std: Any, max_samples: Any, *, maximize: bool = True) -> np.ndarray:
    """Max-value Entropy Search acquisition at candidates with posterior ``mean`` / ``std``.

    Given samples ``max_samples`` of the global optimum value ``y*``, returns the per-candidate mutual
    information ``I(y; y*) = (1/M) sum_m [ gamma_m phi(gamma_m)/(2 Phi(gamma_m)) - log Phi(gamma_m) ]``
    with ``gamma_m = (y*_m - mu)/sd`` (maximization; for minimization the sense is flipped by the
    caller). Higher is better -- it favors uncertain candidates near the believed optimum.
    """
    mu = np.asarray(mean, dtype=np.float64).ravel()
    sd = np.maximum(np.asarray(std, dtype=np.float64).ravel(), 1e-9)
    ystar = np.asarray(max_samples, dtype=np.float64).ravel()
    if not maximize:
        mu, ystar = -mu, -ystar
    gamma = (ystar[None, :] - mu[:, None]) / sd[:, None]  # (n_candidates, M)
    cdf = np.clip(norm.cdf(gamma), 1e-12, 1.0)
    pdf = norm.pdf(gamma)
    info = gamma * pdf / (2.0 * cdf) - np.log(cdf)
    return np.asarray(info.mean(axis=1))


def propose_mes(
    x: Any,
    y: Any,
    bounds: Bounds,
    *,
    n_candidates: int = 512,
    max_samples: int = 64,
    maximize: bool = False,
    seed: int | RandomState | None = None,
    gp: Any = None,
    fit_kwargs: dict[str, Any] | None = None,
) -> np.ndarray:
    """Propose the next point by Max-value Entropy Search.

    Fits the GP to ``(x, y)``, samples ``max_samples`` optimum values via :func:`sample_max_values`,
    scores Latin-hypercube candidates by :func:`max_value_entropy_search`, and returns the maximizer as
    a ``(d,)`` array. ``maximize`` selects the optimization sense (default minimize, matching the rest
    of the BO layer).
    """
    if int(n_candidates) <= 0:
        raise ValueError("n_candidates must be positive.")
    b = _as_bounds(bounds)
    rng = _as_rng(seed)
    xs, ys = _validate_xy(x, y)
    gp = _fit_surrogate(xs, ys, gp, fit_kwargs)
    cand = latin_hypercube(b, int(n_candidates), rng)
    mean, cov = gp.predict(xs, ys, cand, return_cov=True)
    std = np.sqrt(np.clip(np.diag(np.atleast_2d(np.asarray(cov, dtype=np.float64))), 0.0, None))
    # Convert to a maximization of g = +/- f, then run standard MES on g.
    g_mean = np.asarray(mean, dtype=np.float64).ravel() if maximize else -np.asarray(mean, dtype=np.float64).ravel()
    ystar = sample_max_values(g_mean, std, max_samples, seed=int(rng.randint(2**31)))
    merit = max_value_entropy_search(g_mean, std, ystar, maximize=True)
    return cand[int(np.argmax(merit))]


__all__ = ["sample_max_values", "max_value_entropy_search", "propose_mes"]
