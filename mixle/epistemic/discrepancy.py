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

import inspect
from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle.utils.callables import accepts_call

# Keyword names a ``.sample`` implementation may use to accept caller-controlled randomness, in the
# order they are tried. The first three take an RNG object; ``seed`` takes an integer derived from it.
_RNG_KEYWORDS = ("rng", "random_state", "generator", "seed")


def _rng(seed: int | np.random.RandomState | None) -> np.random.RandomState:
    return seed if isinstance(seed, np.random.RandomState) else np.random.RandomState(seed)


def _rng_keyword(fn: Any) -> str | None:
    """The randomness-control keyword ``fn`` accepts, or ``None`` if it exposes none.

    Signature introspection failing (builtins, C extensions) is treated as "no keyword": unlike
    :func:`~mixle.utils.callables.accepts_call`, the safe fallback here is the *reduced* call, since
    passing an ``rng=`` a callable cannot accept would raise at draw time rather than degrade.
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return None
    for name in _RNG_KEYWORDS:
        param = params.get(name)
        if param is not None and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
            return name
    return None


def _is_rng_controlled(dist: Any) -> bool:
    """Whether sampling from ``dist`` can be made reproducible by the RNG this module controls.

    A direct ``.sample(n)`` that accepts no randomness keyword draws from wherever it likes --
    typically NumPy's global RNG -- so a caller-supplied seed does not reach it and two "identically
    seeded" reports genuinely differ. That is not something this module can fix on the object's
    behalf; :func:`discrepancy_report` reports it instead of overclaiming reproducibility.
    """
    direct = getattr(dist, "sample", None)
    if callable(direct):
        return _rng_keyword(direct) is not None
    return callable(getattr(dist, "sampler", None))


def _require_n_draws(samples: np.ndarray, n: int, dist: Any) -> np.ndarray:
    """Enforce that a sampler actually produced the ``n`` draws that were asked for.

    Nothing checked this before, so a sampler that always returned two values satisfied a requested
    budget of 10,000 and the resulting :class:`DiscrepancyResult` still published
    ``n_samples=10000``. That makes the report's own evidence about how much work backed the estimate
    false, and it silently changes the experiment (two draws is a different, far noisier estimator
    than ten thousand). It also lets unequal-length sample arrays reach the pairwise metrics below,
    where NumPy broadcasting turns them into a plausible-looking number rather than an error.
    """
    if samples.shape[0] != n:
        raise ValueError(
            f"{type(dist).__name__} returned {samples.shape[0]} draw(s) for a requested sample budget of "
            f"{n}; a sampler must produce exactly the number of draws it is asked for."
        )
    return samples


def _sample(dist: Any, n: int, rng: np.random.RandomState) -> np.ndarray:
    """Draw exactly ``n`` samples from ``dist``, under this module's RNG wherever that is possible.

    Supports a direct ``.sample(n)`` and mixle's ``.sampler(seed).sample(n)`` shape. When the direct
    ``.sample`` accepts a randomness keyword (see :data:`_RNG_KEYWORDS`) it is threaded through, so a
    seeded call really is seeded; when it does not, the draw is uncontrolled and
    :func:`_is_rng_controlled` reports that to :func:`discrepancy_report` rather than letting the
    result claim a reproducibility it does not have.
    """
    direct = getattr(dist, "sample", None)
    if callable(direct):
        keyword = _rng_keyword(direct)
        if keyword is not None:
            control = int(rng.randint(0, 2**31 - 1)) if keyword == "seed" else rng
            drawn = direct(n, **{keyword: control})
        elif accepts_call(direct, n):
            drawn = direct(n)
        else:
            drawn = [direct() for _ in range(n)]
        return _require_n_draws(np.atleast_1d(np.asarray(drawn, dtype=np.float64)), n, dist)
    sampler_fn = getattr(dist, "sampler", None)
    if callable(sampler_fn):
        seed = int(rng.randint(0, 2**31 - 1))
        drawn = np.atleast_1d(np.asarray(sampler_fn(seed=seed).sample(n), dtype=np.float64))
        return _require_n_draws(drawn, n, dist)
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


def _as_1d_samples(samples: np.ndarray) -> np.ndarray:
    """Squeeze a per-draw sample array down to plain 1D, rejecting genuinely multivariate input.

    A distribution's ``.sample(n)`` can return either a flat ``(n,)`` array or a column-shaped
    ``(n, 1)`` array (one row per draw). The two must be treated identically here, but ``np.sort``
    defaults to sorting along the *last* axis -- for ``(n, 1)`` that axis has exactly one element, so
    sorting it in place is a silent no-op: each row keeps its original draw order, and only *looks*
    sorted once raveled afterwards. Squeezing to 1D here, before the caller sorts, avoids that trap.
    A genuinely multi-column array (more than one value per draw) is rejected outright rather than
    silently flattened -- that is not 1D sample data, and there is no correct way to guess what the
    caller meant by it.
    """
    arr = np.asarray(samples, dtype=np.float64)
    per_draw = arr.reshape(arr.shape[0], -1)
    if per_draw.shape[1] > 1:
        raise NotImplementedError(
            "wasserstein_distance only supports 1D distributions; no cheap exact multivariate "
            "estimator is implemented here (a wrong coordinate-wise number would be worse than refusing)."
        )
    return per_draw.reshape(-1)


def wasserstein_distance(p: Any, q: Any, *, n: int = 10_000, seed: int | np.random.RandomState | None = None) -> float:
    """1-Wasserstein (earth-mover) distance between two 1D distributions, via sorted sample matching.

    Draws ``n`` samples from each side; the empirical 1D optimal transport cost is the mean absolute
    difference between the two sorted sample sequences (exact for the empirical distributions, a
    consistent estimator of the true distance as ``n`` grows). Samples are squeezed to plain 1D
    *before* sorting (:func:`_as_1d_samples`) rather than sorted-then-raveled -- sorting a ``(n, 1)``
    column vector in its original shape is a silent no-op (see that function's docstring), which would
    silently return the mean absolute difference between the two *unsorted* sample sequences instead
    of the true empirical Wasserstein distance. Raises :class:`NotImplementedError` for genuinely
    multivariate input rather than silently computing a coordinate-wise number that isn't the true
    multivariate Wasserstein distance -- there is no cheap exact estimator for that case, and returning
    a wrong-but-plausible-looking number would be worse than refusing.
    """
    rng = _rng(seed)
    xs = _as_1d_samples(_sample(p, n, rng))
    ys = _as_1d_samples(_sample(q, n, rng))
    if xs.shape != ys.shape:
        # Sorted-sample matching pairs the i-th order statistic of each side, which only means
        # anything for two empirical measures of the same size. NumPy would happily broadcast e.g.
        # lengths 2 and 1 into a mean absolute difference and return a plausible number for a
        # comparison that was never made.
        raise ValueError(
            f"wasserstein_distance needs the same number of draws on both sides, got {xs.shape[0]} and "
            f"{ys.shape[0]}; unequal empirical measures cannot be matched by sorted order statistics."
        )
    return float(np.mean(np.abs(np.sort(xs) - np.sort(ys))))


def mmd_squared(
    samples_p: np.ndarray, samples_q: np.ndarray, *, kernel: str = "rbf", bandwidth: float | None = None
) -> float:
    """The **unbiased estimator of squared** MMD between two raw sample sets -- a SIGNED quantity.

    This is Gretton et al.'s U-statistic, and unbiasedness is exactly why it can come out negative:
    under the null (both sample sets from the same distribution) the true value is ``0``, so an
    unbiased estimator of it must scatter to both sides. Identical arrays ``[0, 1]`` give
    ``-0.3934693403``; identical singletons give ``-2.0``. Those are correct outputs of this
    estimator, not errors -- which is why it is named for what it estimates. Use it when you want the
    unbiased test statistic (a two-sample test threshold, a permutation null); use :func:`mmd` when
    you want a discrepancy that behaves like a distance.

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
    if x.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError("mmd requires at least one sample on each side, got %d and %d." % (x.shape[0], y.shape[0]))
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError(f"bandwidth must be positive, got {bandwidth!r}.")
    if bandwidth is None:
        pooled = np.vstack([x, y])
        diffs = pooled[:, None, :] - pooled[None, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=-1))
        nonzero = dists[dists > 0]
        bandwidth = float(np.median(nonzero)) if nonzero.size else 1.0
    gamma = 1.0 / (2.0 * bandwidth**2)

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


def mmd(samples_p: np.ndarray, samples_q: np.ndarray, *, kernel: str = "rbf", bandwidth: float | None = None) -> float:
    """Maximum Mean Discrepancy between two raw sample sets: ``sqrt(max(mmd_squared(...), 0))``.

    :func:`mmd_squared` estimates *squared* MMD and is deliberately signed, so returning it under the
    name ``mmd`` published a "distance" that could be negative -- identical inputs scored below zero,
    and because a discrepancy feeds
    :class:`~mixle.epistemic.likelihood.DiscrepancyLikelihood`'s ``exp(-discrepancy / temperature)``,
    a negative value produced a likelihood above ``1`` and reversed the intended penalty ordering
    right around zero, where the two distributions are most alike.

    Taking the square root of the clipped estimate restores the contract the public discrepancy API
    is written against: non-negative, zero when the samples are indistinguishable, growing with
    separation, and in the same units as the data. The clip means this is *not* unbiased near zero
    (it can only overstate a true zero, never understate it); when unbiasedness is what matters --
    two-sample testing, permutation nulls -- call :func:`mmd_squared` directly and handle its sign
    yourself.
    """
    estimate = mmd_squared(samples_p, samples_q, kernel=kernel, bandwidth=bandwidth)
    return float(np.sqrt(max(estimate, 0.0)))


@dataclass(frozen=True)
class DiscrepancyResult:
    """One discrepancy evaluation: the value, which metric computed it, and whether it was exact.

    ``seed`` and ``n_samples`` record what a Monte Carlo estimate needs for exact reproduction: pass
    ``seed`` back into :func:`discrepancy_report` and the same ``value`` comes back out. Both are
    ``None`` when nothing was sampled -- an exact closed form, or a metric applied directly to
    caller-supplied arrays -- since recording a seed there would misleadingly imply randomness that
    was never used. ``seed`` is also ``None`` when the caller passed their own
    ``np.random.RandomState`` instance rather than an int: that object's reproducibility is already
    the caller's to manage, and there is no single integer that reconstructs it.

    ``reproducible`` is ``False`` when a sampled side draws from randomness this module cannot
    control -- a bare ``.sample(n)`` that takes no RNG/seed argument and reads NumPy's global state.
    A seed passed to such a call never reaches the sampler, so two identically seeded reports return
    genuinely different values; ``seed`` is therefore also ``None`` in that case rather than
    recording an integer that does not reproduce anything. Give the sampler an ``rng`` /
    ``random_state`` / ``generator`` / ``seed`` keyword, or expose mixle's ``.sampler(seed)`` shape,
    to make a path reproducible.
    """

    value: float
    metric: str
    degraded: bool
    seed: int | None = None
    n_samples: int | None = None
    reproducible: bool = True


def _resolve_report_seed(seed: int | np.random.RandomState | None) -> tuple[np.random.RandomState, int | None]:
    """Resolve ``discrepancy_report``'s ``seed`` argument to an RNG plus the integer to record.

    An explicit int is used as-is and echoed back unchanged. ``None`` means the caller didn't ask for
    a specific seed -- rather than seeding from OS entropy and leaving that draw forever
    unrecoverable, a fresh integer is generated *first*, used to build the RNG, and handed back so it
    can be recorded on the result: an unseeded call must still be reproducible afterwards by whoever
    receives that result, even though they didn't pick the seed themselves. An explicit
    ``RandomState`` instance is used as-is (mirroring :func:`_rng`); there is no single integer that
    reconstructs it, so the recorded seed is ``None`` in that case -- the caller already owns that
    object's reproducibility.
    """
    if isinstance(seed, np.random.RandomState):
        return seed, None
    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1)[0])
    return np.random.RandomState(seed), int(seed)


def _resolve_sampling(
    seed: int | np.random.RandomState | None, *sampled: Any
) -> tuple[np.random.RandomState, int | None, bool]:
    """RNG, recordable seed, and reproducibility for a report path that draws from ``sampled``.

    Splits "which RNG do we drive" from "can we honestly promise this run reproduces". The second
    question is about the objects being sampled, not about the seed: if any of them ignores the RNG
    we hand it (see :func:`_is_rng_controlled`), the recorded seed would be a claim the report cannot
    back, so it is dropped along with the ``reproducible`` flag.
    """
    rng, seed_used = _resolve_report_seed(seed)
    reproducible = all(_is_rng_controlled(dist) for dist in sampled)
    return rng, (seed_used if reproducible else None), reproducible


# Sample budgets for discrepancy_report's own Monte Carlo paths, named so DiscrepancyResult.n_samples
# has one source of truth instead of a literal repeated at every call site below.
_MMD_PROXY_SAMPLES = 512  # predicted-vs-observed-array mmd path: samples drawn from `predicted`
_DEFAULT_MC_N = 10_000  # mirrors kl_divergence/js_divergence/wasserstein_distance's own `n` default


def discrepancy_report(
    predicted: Any, observed: Any, *, metric: str = "auto", seed: int | np.random.RandomState | None = None
) -> DiscrepancyResult:
    """The actual ``delta_m(o_hat, o)`` entry point: compare a predicted and an observed value/distribution.

    ``metric="auto"`` picks ``kl_divergence`` when both sides look like distributions (expose
    ``log_density``), else ``mmd`` over raw arrays (the "predicted is a distribution, observed is a
    concrete measurement" case reduces to comparing ``observed`` against samples drawn from
    ``predicted``). ``degraded=True`` whenever the underlying computation fell back to a Monte Carlo /
    sample-based estimate rather than an exact closed form -- callers that need to know whether a
    number is exact or estimated read this field rather than guessing from the metric name.

    Every path that draws samples does so from a seeded RNG, never from unrecorded OS entropy: pass
    ``seed`` for a run you choose to make reproducible, or leave it unset and read the seed this
    function picked back off ``result.seed`` -- either way, calling again with that same ``seed``
    reproduces the exact same ``value``, PROVIDED the sampled objects let this module control their
    randomness. When one does not, ``result.reproducible`` is ``False`` and ``result.seed`` is
    ``None``: the guarantee is reported as unavailable rather than quietly broken. See
    :class:`DiscrepancyResult` for that contract and for when ``n_samples`` comes back ``None``.
    """
    if metric == "auto":
        predicted_is_dist = callable(getattr(predicted, "log_density", None))
        observed_is_dist = callable(getattr(observed, "log_density", None))
        if predicted_is_dist and observed_is_dist:
            exact = _is_univariate_gaussian(predicted) and _is_univariate_gaussian(observed)
            if exact:
                return DiscrepancyResult(kl_divergence(predicted, observed), "kl_divergence", degraded=False)
            rng, seed_used, repro = _resolve_sampling(seed, predicted)
            value = kl_divergence(predicted, observed, seed=rng)
            return DiscrepancyResult(
                value, "kl_divergence", degraded=True, seed=seed_used, n_samples=_DEFAULT_MC_N, reproducible=repro
            )
        if predicted_is_dist and not observed_is_dist:
            rng, seed_used, repro = _resolve_sampling(seed, predicted)
            pred_samples = _sample(predicted, _MMD_PROXY_SAMPLES, rng)
            obs_samples = np.atleast_1d(np.asarray(observed, dtype=np.float64))
            value = mmd(pred_samples, obs_samples)
            return DiscrepancyResult(
                value, "mmd", degraded=True, seed=seed_used, n_samples=_MMD_PROXY_SAMPLES, reproducible=repro
            )
        pred_arr = np.atleast_1d(np.asarray(predicted, dtype=np.float64))
        obs_arr = np.atleast_1d(np.asarray(observed, dtype=np.float64))
        return DiscrepancyResult(mmd(pred_arr, obs_arr), "mmd", degraded=True)
    if metric == "kl_divergence":
        exact = _is_univariate_gaussian(predicted) and _is_univariate_gaussian(observed)
        if exact:
            return DiscrepancyResult(kl_divergence(predicted, observed), metric, degraded=False)
        rng, seed_used, repro = _resolve_sampling(seed, predicted)
        value = kl_divergence(predicted, observed, seed=rng)
        return DiscrepancyResult(
            value, metric, degraded=True, seed=seed_used, n_samples=_DEFAULT_MC_N, reproducible=repro
        )
    if metric == "js_divergence":
        rng, seed_used, repro = _resolve_sampling(seed, predicted, observed)
        value = js_divergence(predicted, observed, seed=rng)
        return DiscrepancyResult(
            value, metric, degraded=True, seed=seed_used, n_samples=_DEFAULT_MC_N, reproducible=repro
        )
    if metric == "wasserstein_distance":
        rng, seed_used, repro = _resolve_sampling(seed, predicted, observed)
        value = wasserstein_distance(predicted, observed, seed=rng)
        return DiscrepancyResult(
            value, metric, degraded=True, seed=seed_used, n_samples=_DEFAULT_MC_N, reproducible=repro
        )
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
    "mmd_squared",
]
