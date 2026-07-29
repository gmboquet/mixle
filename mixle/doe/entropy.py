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

from mixle.doe.batch import _safe_cholesky
from mixle.doe.bayesopt import _fit_surrogate, _validate_xy
from mixle.doe.designs import Bounds, _as_bounds, _as_rng, _require_exact_positive_int, latin_hypercube

# y_r = a - b * log(-log r) for the Gumbel; constants at r = 0.25, 0.5, 0.75.
_C25, _C50, _C75 = float(np.log(-np.log(0.25))), float(np.log(-np.log(0.5))), float(np.log(-np.log(0.75)))


def _require_positive_int(value: Any, name: str) -> int:
    """Validate ``value`` is an exact, finite, positive integer count.

    A bare ``int(value)`` truncates fractional counts and accepts non-positive ones silently -- e.g. a
    nonpositive ``n_samples`` would otherwise draw a silent empty ``y*`` sample with no error
    (MXR-080-0178). This names both as invalid instead.
    """
    return _require_exact_positive_int(value, name)


def _validate_moments(mean: Any, std: Any) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return posterior ``(mean, std)`` as matching, non-empty, finite 1-D arrays.

    MXR-080-0178: an empty ``mean``/``std`` previously reached the Gumbel fit's ``.min()``/``.max()``
    calls and crashed with an opaque "zero-size array" error; non-finite moments silently propagated to
    NaN acquisition values; and a negative ``std`` (never legitimate -- a standard deviation cannot be
    negative, so this means corrupted upstream input, not a numerical-precision edge case) was floored
    to the same ``1e-9`` epsilon used for a legitimately tiny-but-valid std instead of being rejected.
    This rejects all of those up front and floors only genuinely nonnegative std, for numerical
    stability in the divisions/logs that follow.
    """
    mu = np.asarray(mean, dtype=np.float64).ravel()
    sd = np.asarray(std, dtype=np.float64).ravel()
    if mu.size == 0:
        raise ValueError("mean must have at least one candidate (got an empty array).")
    if sd.shape != mu.shape:
        raise ValueError(f"std must have the same shape as mean {mu.shape}; got {sd.shape}.")
    if not np.all(np.isfinite(mu)):
        raise ValueError("mean contains non-finite values (NaN/Inf).")
    if not np.all(np.isfinite(sd)):
        raise ValueError("std contains non-finite values (NaN/Inf).")
    if np.any(sd < 0.0):
        raise ValueError("std must be nonnegative (a standard deviation cannot be negative).")
    return mu, np.maximum(sd, 1e-9)


def _gumbel_quantile(loc: float, scale: float, r: Any) -> np.ndarray:
    """Quantile function of a Gumbel(``loc``, ``scale``): ``y_r = loc - scale * log(-log(r))``.

    The standard Gumbel (maxima convention) has CDF ``F(z) = exp(-exp(-z))``; solving ``r = F(z)`` for
    ``z`` gives ``z = -log(-log(r))``. A general Gumbel(loc, scale) draw is ``Y = loc + scale*Z`` for
    standard Gumbel ``Z``, so its quantile function is
    ``y_r = loc + scale*(-log(-log(r))) = loc - scale*log(-log(r)) = loc - scale*C_r``, with
    ``C_r := log(-log(r))`` (the module-level ``_C25``/``_C50``/``_C75`` constants).
    """
    return loc - scale * np.log(-np.log(np.asarray(r, dtype=np.float64)))


def _fit_gumbel(y25: float, y50: float, y75: float) -> tuple[float, float]:
    """Fit Gumbel(loc, scale) to target 25/50/75 percentiles by quantile matching.

    From ``y_r = loc - scale*C_r`` (:func:`_gumbel_quantile`):

    * ``scale`` from the IQR: ``y75 - y25 = -scale*C75 - (-scale*C25) = scale*(C25 - C75)``, so
      ``scale = (y75 - y25) / (C25 - C75)`` (``C25 > C75``, so this is positive; floored at ``1e-6``).
    * ``loc`` from the median: ``y50 = loc - scale*C50`` solves to ``loc = y50 + scale*C50``.

    MXR-080-0177: this previously SUBTRACTED here (``loc = y50 - scale*C50``). Plugging the wrong
    ``loc`` back into the quantile function at ``r=0.5`` gives a fitted Gumbel whose own median is
    ``loc - scale*C50 = y50 - 2*scale*C50``, not ``y50`` -- off by ``-2*scale*C50``. For one
    standard-normal candidate (``y50 = 0``), that is about ``+0.63`` (``C50 = log(-log(0.5)) ~= -0.3665``
    and the fitted ``scale ~= 0.858`` there), matching the audit's reproduction. The inverse-CDF test
    for this function fits against known quantiles of a reference Gumbel and confirms both parameters
    -- and the recovered median specifically -- come back exact.
    """
    scale = max((y75 - y25) / (_C25 - _C75), 1e-6)  # b from the IQR
    loc = y50 + scale * _C50  # a from the median
    return loc, scale


def sample_max_values(mean: Any, std: Any, n_samples: int = 64, *, seed: int | RandomState = 0) -> np.ndarray:
    """Sample plausible global-max values ``y*`` via the Gumbel approximation (Wang & Jegelka 2017).

    The CDF of the maximum over the candidate cloud is ``P(max <= y) = prod_i Phi((y - mu_i)/sd_i)``; a
    Gumbel is fit to its 25/50/75 percentiles (found by bisection) and sampled. Returns an
    ``(n_samples,)`` array of ``y*`` draws (never below the best posterior mean).

    Raises:
        ValueError: if ``mean``/``std`` are empty, mismatched in shape, non-finite, or ``std`` is
            negative; or if ``n_samples`` is not a positive integer.
    """
    mu, sd = _validate_moments(mean, std)
    n = _require_positive_int(n_samples, "n_samples")
    rng = seed if isinstance(seed, RandomState) else RandomState(seed)

    def cdf_max(y: float) -> float:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            standardized = (y - mu) / sd
        if not np.all(np.isfinite(standardized)):
            raise ValueError("MES maximum-value search produced a non-finite standardized bracket.")
        result = float(np.exp(np.sum(norm.logcdf(standardized))))
        if not np.isfinite(result):
            raise ValueError("MES maximum-value CDF became non-finite.")
        return result

    with np.errstate(over="ignore", invalid="ignore"):
        lo = float((mu - 5.0 * sd).min())
        hi = float((mu + 8.0 * sd).max())
    if not np.isfinite(lo) or not np.isfinite(hi) or lo > hi:
        raise ValueError("MES maximum-value search bracket is not representable as finite float64 values.")

    def quantile(target: float) -> float:
        a, b = lo, hi
        for _ in range(60):
            m = 0.5 * a + 0.5 * b
            if cdf_max(m) < target:
                a = m
            else:
                b = m
        result = 0.5 * a + 0.5 * b
        if not np.isfinite(result):
            raise ValueError("MES maximum-value quantile became non-finite.")
        return result

    y25, y50, y75 = quantile(0.25), quantile(0.5), quantile(0.75)
    loc, scale = _fit_gumbel(y25, y50, y75)
    if not np.isfinite(loc) or not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("MES Gumbel fit produced non-finite or non-positive parameters.")
    u = rng.uniform(1e-6, 1.0 - 1e-6, n)
    with np.errstate(over="ignore", invalid="ignore"):
        ystar = np.maximum(_gumbel_quantile(loc, scale, u), mu.max())
    if not np.all(np.isfinite(ystar)):
        raise ValueError("MES maximum-value samples became non-finite.")
    return ystar


def max_value_entropy_search(mean: Any, std: Any, max_samples: Any, *, maximize: bool = True) -> np.ndarray:
    """Max-value Entropy Search acquisition at candidates with posterior ``mean`` / ``std``.

    Given samples ``max_samples`` of the global optimum value ``y*``, returns the per-candidate mutual
    information ``I(y; y*) = (1/M) sum_m [ gamma_m phi(gamma_m)/(2 Phi(gamma_m)) - log Phi(gamma_m) ]``
    with ``gamma_m = (y*_m - mu)/sd`` (maximization; for minimization the sense is flipped by the
    caller). Higher is better -- it favors uncertain candidates near the believed optimum.

    Raises:
        ValueError: if ``mean``/``std`` are empty, mismatched in shape, non-finite, or ``std`` is
            negative; if ``max_samples`` is empty or non-finite; or if the resulting per-candidate
            information is non-finite (e.g. an extreme-but-individually-finite ``mean``/``std``/
            ``max_samples`` combination can overflow ``gamma = (y* - mu)/sd`` to +/-inf, and
            ``inf * 0`` in the ``gamma * pdf`` term then silently yields NaN).
    """
    mu, sd = _validate_moments(mean, std)
    if type(maximize) is not bool:
        raise TypeError(f"maximize must be a bool, got {type(maximize).__name__}.")
    ystar = np.asarray(max_samples, dtype=np.float64).ravel()
    if ystar.size == 0:
        raise ValueError("max_samples must contain at least one optimum-value sample (got an empty array).")
    if not np.all(np.isfinite(ystar)):
        raise ValueError("max_samples contains non-finite values (NaN/Inf).")
    if not maximize:
        mu, ystar = -mu, -ystar
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        gamma = (ystar[None, :] - mu[:, None]) / sd[:, None]  # (n_candidates, M)
    if not np.all(np.isfinite(gamma)):
        raise ValueError("max_value_entropy_search produced non-finite standardized optimum values.")
    cdf = np.clip(norm.cdf(gamma), 1e-12, 1.0)
    pdf = norm.pdf(gamma)
    info = gamma * pdf / (2.0 * cdf) - np.log(cdf)
    result = np.asarray(info.mean(axis=1))
    if not np.all(np.isfinite(result)):
        raise ValueError("max_value_entropy_search produced non-finite information for at least one candidate.")
    return result


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
    n_candidates = _require_exact_positive_int(n_candidates, "n_candidates")
    max_samples = _require_exact_positive_int(max_samples, "max_samples")
    if type(maximize) is not bool:
        raise TypeError(f"maximize must be a bool, got {type(maximize).__name__}.")
    b = _as_bounds(bounds)
    rng = _as_rng(seed)
    xs, ys = _validate_xy(x, y)
    gp = _fit_surrogate(xs, ys, gp, fit_kwargs)
    cand = latin_hypercube(b, n_candidates, rng)
    if cand.shape != (n_candidates, b.shape[0]) or not np.all(np.isfinite(cand)):
        raise ValueError(
            f"latin_hypercube must return a finite ({n_candidates}, {b.shape[0]}) candidate matrix; got {cand.shape}."
        )
    mean_raw, cov_raw = gp.predict(xs, ys, cand, return_cov=True)
    mean = np.asarray(mean_raw, dtype=np.float64)
    if mean.shape == (n_candidates, 1):
        mean = mean[:, 0]
    elif mean.shape != (n_candidates,):
        raise ValueError(
            f"MES posterior mean must have shape ({n_candidates},) or ({n_candidates}, 1), got {mean.shape}."
        )
    if not np.all(np.isfinite(mean)):
        raise ValueError("MES posterior mean must be finite.")
    cov = np.asarray(cov_raw, dtype=np.float64)
    if cov.shape != (n_candidates, n_candidates):
        raise ValueError(f"MES posterior covariance must have shape ({n_candidates}, {n_candidates}), got {cov.shape}.")
    _safe_cholesky(cov)
    variance = np.diag(cov)
    if np.any(variance < 0.0):
        raise ValueError("MES posterior marginal variances must be nonnegative.")
    std = np.sqrt(variance)
    # Convert to a maximization of g = +/- f, then run standard MES on g.
    g_mean = mean if maximize else -mean
    ystar = sample_max_values(g_mean, std, max_samples, seed=int(rng.randint(2**31)))
    merit = max_value_entropy_search(g_mean, std, ystar, maximize=True)
    return cand[int(np.argmax(merit))]


__all__ = ["sample_max_values", "max_value_entropy_search", "propose_mes"]
