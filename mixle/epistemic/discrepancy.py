"""Distance/divergence between two distributions, or between a predicted and an observed sample.

This is the "compare predicted vs. observed" hinge an epistemic loop's UPDATE arrow needs: given a
hypothesis's predicted observation and the real one, how far apart are they? Nothing under
``mixle.stats``/``mixle.inference`` computed this directly before this module -- proper scoring rules
(:mod:`mixle.inference.scoring`) score a single outcome against a predictive distribution, which is a
related but different question (they answer "how good was this one call", not "how far apart are these
two whole distributions").

Every function here is generic over any object exposing ``log_density``/``sample`` (the same duck-typed
surface :mod:`mixle.capability` already dispatches on) or ``.sampler(seed).sample(n)`` (the concrete
shape every :mod:`mixle.stats` distribution actually has). A closed-form fast path is used only where
one is exact and unambiguous (currently: two univariate Gaussians); everything else falls back to a
Monte Carlo estimate, and :func:`discrepancy_report` says plainly which path was taken via its
``degraded`` flag -- an honest signal, never a silently approximated number presented as exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _rng(seed: int | np.random.RandomState | None) -> np.random.RandomState:
    return seed if isinstance(seed, np.random.RandomState) else np.random.RandomState(seed)


def _sample(dist: Any, n: int, rng: np.random.RandomState) -> np.ndarray:
    """Draw ``n`` samples from ``dist``, supporting both a direct ``.sample(n)`` and mixle's
    ``.sampler(seed).sample(n)`` distribution shape."""
    direct = getattr(dist, "sample", None)
    if callable(direct):
        try:
            return np.atleast_1d(np.asarray(direct(n), dtype=np.float64))
        except TypeError:
            return np.atleast_1d(np.asarray([direct() for _ in range(n)], dtype=np.float64))
    sampler_fn = getattr(dist, "sampler", None)
    if callable(sampler_fn):
        seed = int(rng.randint(0, 2**31 - 1))
        return np.atleast_1d(np.asarray(sampler_fn(seed=seed).sample(n), dtype=np.float64))
    raise TypeError(f"{type(dist).__name__} exposes neither .sample(n) nor .sampler(seed).sample(n)")


def _log_density(dist: Any, xs: np.ndarray) -> np.ndarray:
    """Evaluate ``dist``'s log-density at every element of ``xs`` (scalar ``log_density`` looped)."""
    fn = dist.log_density
    return np.array([float(fn(x)) for x in np.atleast_1d(xs)], dtype=np.float64)


def _is_univariate_gaussian(dist: Any) -> bool:
    from mixle.stats.univariate.continuous.gaussian import GaussianDistribution

    return isinstance(dist, GaussianDistribution)


def kl_divergence(p: Any, q: Any, *, n: int = 10_000, seed: int | np.random.RandomState | None = None) -> float:
    """KL(p || q) in nats: exact closed form when a known pair matches, else a Monte Carlo estimate.

    The one closed-form entry in the dispatch table today is two univariate Gaussians (the exact
    formula, not an approximation); every other pair falls back to
    ``mean_{x ~ p}[log p(x) - log q(x)]`` using ``n`` samples drawn from ``p``. Extending the
    closed-form table to more conjugate pairs (Categorical-Categorical, Dirichlet-Dirichlet, ...) is
    legitimate future work -- it was deliberately left at one entry here rather than half-built across
    several families with incompatible parameterizations (mixle's categorical distribution keys its
    simplex by a ``pmap`` over arbitrary hashable labels, not a fixed-order probability vector, which
    is a real complication left to a dedicated follow-up rather than papered over).
    """
    if _is_univariate_gaussian(p) and _is_univariate_gaussian(q):
        mu_p, var_p = float(p.mean()), float(p.variance())
        mu_q, var_q = float(q.mean()), float(q.variance())
        return float(0.5 * (var_p / var_q + (mu_q - mu_p) ** 2 / var_q - 1.0 + np.log(var_q / var_p)))
    rng = _rng(seed)
    xs = _sample(p, n, rng)
    return float(np.mean(_log_density(p, xs) - _log_density(q, xs)))


def js_divergence(p: Any, q: Any, *, n: int = 10_000, seed: int | np.random.RandomState | None = None) -> float:
    """Jensen-Shannon divergence: symmetric, bounded, computed via the sample-mixture estimator.

    ``0.5 * KL(p || m) + 0.5 * KL(q || m)`` where ``m`` is the equal mixture of ``p`` and ``q``; each
    term is estimated by sampling from the corresponding side and evaluating ``log m(x) = log(0.5 p(x)
    + 0.5 q(x))`` via ``logaddexp`` for numerical stability. Symmetric by construction up to Monte
    Carlo noise (both halves use independent sample draws).
    """
    rng = _rng(seed)
    half = max(1, n // 2)

    def _half_kl_to_mixture(source: Any) -> float:
        xs = _sample(source, half, rng)
        log_src = _log_density(source, xs)
        log_p = _log_density(p, xs)
        log_q = _log_density(q, xs)
        log_mix = np.logaddexp(log_p, log_q) - np.log(2.0)
        return float(np.mean(log_src - log_mix))

    return float(0.5 * _half_kl_to_mixture(p) + 0.5 * _half_kl_to_mixture(q))


def wasserstein_distance(p: Any, q: Any, *, n: int = 10_000, seed: int | np.random.RandomState | None = None) -> float:
    """1-Wasserstein (earth-mover) distance between two 1D distributions, via sorted sample matching.

    Draws ``n`` samples from each side; the empirical 1D optimal transport cost is the mean absolute
    difference between the two sorted sample sequences (exact for the empirical distributions, a
    consistent estimator of the true distance as ``n`` grows). Raises :class:`NotImplementedError` for
    multivariate input rather than silently computing a coordinate-wise number that isn't the true
    multivariate Wasserstein distance -- there is no cheap exact estimator for that case, and returning
    a wrong-but-plausible-looking number would be worse than refusing.
    """
    rng = _rng(seed)
    xs = np.sort(_sample(p, n, rng))
    ys = np.sort(_sample(q, n, rng))
    if xs.ndim > 1 and xs.shape[-1] > 1:
        raise NotImplementedError(
            "wasserstein_distance only supports 1D distributions; no cheap exact multivariate "
            "estimator is implemented here (a wrong coordinate-wise number would be worse than refusing)."
        )
    return float(np.mean(np.abs(xs.ravel() - ys.ravel())))


def mmd(samples_p: np.ndarray, samples_q: np.ndarray, *, kernel: str = "rbf", bandwidth: float | None = None) -> float:
    """Maximum Mean Discrepancy between two raw sample sets (unbiased estimator).

    Unlike the other functions here, this takes samples directly rather than distribution objects --
    it works even when neither side is a ``mixle.stats`` distribution (e.g. a real observation array
    vs. a synthesized/predicted one). ``bandwidth`` defaults to the median pairwise distance heuristic
    over the pooled samples. Only the RBF kernel is implemented; other kernel names raise
    :class:`NotImplementedError`.
    """
    if kernel != "rbf":
        raise NotImplementedError(f"mmd only implements the 'rbf' kernel, got {kernel!r}")

    def _prepare(a: np.ndarray) -> np.ndarray:
        arr = np.asarray(a, dtype=np.float64)
        return arr.reshape(-1, 1) if arr.ndim == 1 else arr

    x = _prepare(samples_p)
    y = _prepare(samples_q)
    if bandwidth is None:
        pooled = np.vstack([x, y])
        diffs = pooled[:, None, :] - pooled[None, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=-1))
        nonzero = dists[dists > 0]
        bandwidth = float(np.median(nonzero)) if nonzero.size else 1.0
    gamma = 1.0 / (2.0 * bandwidth**2) if bandwidth > 0 else 1.0

    def _rbf(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        diffs = a[:, None, :] - b[None, :, :]
        return np.exp(-gamma * np.sum(diffs**2, axis=-1))

    kxx = _rbf(x, x)
    kyy = _rbf(y, y)
    kxy = _rbf(x, y)
    m, n = x.shape[0], y.shape[0]
    term_xx = (kxx.sum() - np.trace(kxx)) / (m * (m - 1)) if m > 1 else 0.0
    term_yy = (kyy.sum() - np.trace(kyy)) / (n * (n - 1)) if n > 1 else 0.0
    term_xy = kxy.sum() / (m * n)
    return float(term_xx + term_yy - 2.0 * term_xy)


@dataclass(frozen=True)
class DiscrepancyResult:
    """One discrepancy evaluation: the value, which metric computed it, and whether it was exact."""

    value: float
    metric: str
    degraded: bool


def discrepancy_report(predicted: Any, observed: Any, *, metric: str = "auto") -> DiscrepancyResult:
    """The actual ``delta_m(o_hat, o)`` entry point: compare a predicted and an observed value/distribution.

    ``metric="auto"`` picks ``kl_divergence`` when both sides look like distributions (expose
    ``log_density``), else ``mmd`` over raw arrays (the "predicted is a distribution, observed is a
    concrete measurement" case reduces to comparing ``observed`` against samples drawn from
    ``predicted``). ``degraded=True`` whenever the underlying computation fell back to a Monte Carlo /
    sample-based estimate rather than an exact closed form -- callers that need to know whether a
    number is exact or estimated read this field rather than guessing from the metric name.
    """
    if metric == "auto":
        predicted_is_dist = callable(getattr(predicted, "log_density", None))
        observed_is_dist = callable(getattr(observed, "log_density", None))
        if predicted_is_dist and observed_is_dist:
            exact = _is_univariate_gaussian(predicted) and _is_univariate_gaussian(observed)
            return DiscrepancyResult(kl_divergence(predicted, observed), "kl_divergence", degraded=not exact)
        if predicted_is_dist and not observed_is_dist:
            rng = _rng(None)
            pred_samples = _sample(predicted, 512, rng)
            obs_samples = np.atleast_1d(np.asarray(observed, dtype=np.float64))
            return DiscrepancyResult(mmd(pred_samples, obs_samples), "mmd", degraded=True)
        pred_arr = np.atleast_1d(np.asarray(predicted, dtype=np.float64))
        obs_arr = np.atleast_1d(np.asarray(observed, dtype=np.float64))
        return DiscrepancyResult(mmd(pred_arr, obs_arr), "mmd", degraded=True)
    if metric == "kl_divergence":
        exact = _is_univariate_gaussian(predicted) and _is_univariate_gaussian(observed)
        return DiscrepancyResult(kl_divergence(predicted, observed), metric, degraded=not exact)
    if metric == "js_divergence":
        return DiscrepancyResult(js_divergence(predicted, observed), metric, degraded=True)
    if metric == "wasserstein_distance":
        return DiscrepancyResult(wasserstein_distance(predicted, observed), metric, degraded=True)
    if metric == "mmd":
        return DiscrepancyResult(mmd(predicted, observed), metric, degraded=True)
    raise ValueError(f"unknown metric {metric!r}")


__all__ = [
    "DiscrepancyResult",
    "discrepancy_report",
    "kl_divergence",
    "js_divergence",
    "wasserstein_distance",
    "mmd",
]
