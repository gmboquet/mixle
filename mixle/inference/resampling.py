"""Bootstrap and permutation inference for arbitrary statistics.

Distribution-free uncertainty estimates by resampling the data itself:

  * :func:`bootstrap` -- a confidence interval for *any* statistic ``T(data)``, with the resampling
    scheme matched to the data's dependence structure: plain i.i.d., **stratified** (resample within
    groups), **cluster/hierarchical** (resample whole clusters -- the unit of independence), **moving
    block** (preserve autocorrelation in a series), or **m-out-of-n subsampling**. Interval types:
    ``percentile``, ``basic`` (pivotal), and ``bca`` (bias-corrected and accelerated -- the
    second-order-accurate default for i.i.d. data).
  * :func:`wild_bootstrap` -- residual bootstrap for regression that is robust to heteroscedasticity
    (Rademacher or Mammen two-point multipliers on the residuals).
  * :func:`permutation_test` -- an exact/Monte-Carlo test for an arbitrary statistic under a sharp
    null, with **stratified / restricted** (within-group) shuffling and a **paired** (sign-flip) mode;
    when the number of distinct rearrangements is small it enumerates them for an *exact* p-value.

Everything is pure NumPy. ``data`` may be a single array (resampled along axis 0) or a tuple of arrays
sharing their first axis (e.g. ``(X, y)``); the statistic is then called as ``statistic(*parts)``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.stats import norm


def _as_rng(seed: int | RandomState | None) -> RandomState:
    if isinstance(seed, RandomState):
        return seed
    return RandomState(seed)


def _is_tuple(data: Any) -> bool:
    return isinstance(data, (tuple, list))


def _n_units(data: Any) -> int:
    return len(data[0]) if _is_tuple(data) else len(data)


def _take(data: Any, idx: np.ndarray) -> Any:
    if _is_tuple(data):
        return tuple(np.asarray(d)[idx] for d in data)
    return np.asarray(data)[idx]


def _call(statistic: Callable, data: Any) -> np.ndarray:
    out = statistic(*data) if _is_tuple(data) else statistic(data)
    return np.asarray(out, dtype=float)


@dataclass
class BootstrapResult:
    """Result of a :func:`bootstrap` call.

    Attributes:
        estimate: the statistic on the original data (scalar or vector).
        ci_low / ci_high: confidence-interval endpoints (same shape as ``estimate``).
        distribution: ``(n_boot, ...)`` array of bootstrap replicates.
        method: the interval method used.
        ci_level: the central probability of the interval.
        standard_error: bootstrap standard error (std of the replicates).
    """

    estimate: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    distribution: np.ndarray
    method: str
    ci_level: float
    standard_error: np.ndarray


def _resample_indices(
    n: int,
    rng: RandomState,
    *,
    groups: np.ndarray | None,
    clusters: np.ndarray | None,
    block_length: int | None,
    m: int | None,
) -> np.ndarray:
    """Draw one resample's row indices under the requested scheme."""
    if clusters is not None:
        labels = np.unique(clusters)
        chosen = rng.choice(labels, size=len(labels), replace=True)
        return np.concatenate([np.nonzero(clusters == c)[0] for c in chosen])
    if groups is not None:
        idx = np.empty(n, dtype=int)
        for g in np.unique(groups):
            pos = np.nonzero(groups == g)[0]
            idx[pos] = rng.choice(pos, size=len(pos), replace=True)
        return idx
    if block_length is not None:
        if not 1 <= block_length <= n:
            raise ValueError("block_length must be in [1, n].")
        n_blocks = int(np.ceil(n / block_length))
        starts = rng.randint(0, n - block_length + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_length) for s in starts])
        return idx[:n]
    if m is not None:
        return rng.choice(n, size=m, replace=False)
    return rng.randint(0, n, size=n)


def bootstrap(
    data: Any,
    statistic: Callable[..., Any],
    *,
    n_boot: int = 2000,
    method: str = "bca",
    ci_level: float = 0.95,
    seed: int | RandomState | None = 0,
    groups: np.ndarray | None = None,
    clusters: np.ndarray | None = None,
    block_length: int | None = None,
    m: int | None = None,
) -> BootstrapResult:
    """Bootstrap confidence interval for ``statistic(data)``.

    Args:
        data: a single array (resampled along axis 0) or a tuple of arrays sharing their first axis
            (the statistic is then called as ``statistic(*parts)``).
        statistic: maps the data to a scalar or fixed-length vector.
        n_boot: number of bootstrap resamples.
        method: ``"percentile"``, ``"basic"`` (pivotal), or ``"bca"`` (bias-corrected & accelerated).
            ``"bca"`` is only second-order accurate for plain i.i.d. resampling; with ``groups`` /
            ``clusters`` / ``block_length`` / ``m`` set it falls back to ``"percentile"``.
        ci_level: central probability of the interval.
        seed: RNG seed.
        groups: ``(n,)`` labels for **stratified** resampling (resample within each group).
        clusters: ``(n,)`` labels for **cluster** resampling (resample whole clusters with replacement).
        block_length: moving-**block** length for serially dependent (time-series) data.
        m: subsample size for **m-out-of-n** subsampling (without replacement). Replicates are
            rescaled about the point estimate by ``sqrt(m/n)`` (Politis--Romano), so the returned
            distribution / interval / standard error are at full-sample scale.

    Returns:
        A :class:`BootstrapResult`.
    """
    rng = _as_rng(seed)
    n = _n_units(data)
    estimate = _call(statistic, data)
    reps = np.empty((n_boot,) + estimate.shape, dtype=float)
    special = groups is not None or clusters is not None or block_length is not None or m is not None
    for b in range(n_boot):
        idx = _resample_indices(n, rng, groups=groups, clusters=clusters, block_length=block_length, m=m)
        reps[b] = _call(statistic, _take(data, idx))
    if m is not None and m < n:
        # Politis-Romano m-out-of-n rescaling: a size-m subsample statistic fluctuates at the
        # sqrt(m) rate, so shrink the replicates about the point estimate by sqrt(m/n) to put
        # them at full-sample scale (assumes the usual sqrt(n)-consistent statistic).
        reps = estimate + np.sqrt(m / n) * (reps - estimate)

    alpha = 1.0 - ci_level
    if method == "bca" and special:
        method_used = "percentile"
    else:
        method_used = method

    if method_used == "percentile":
        lo = np.quantile(reps, alpha / 2.0, axis=0)
        hi = np.quantile(reps, 1.0 - alpha / 2.0, axis=0)
    elif method_used == "basic":
        lo = 2.0 * estimate - np.quantile(reps, 1.0 - alpha / 2.0, axis=0)
        hi = 2.0 * estimate - np.quantile(reps, alpha / 2.0, axis=0)
    elif method_used == "bca":
        lo, hi = _bca_interval(data, statistic, estimate, reps, alpha)
    else:
        raise ValueError("method must be 'percentile', 'basic', or 'bca'.")

    return BootstrapResult(
        estimate=estimate,
        ci_low=np.asarray(lo),
        ci_high=np.asarray(hi),
        distribution=reps,
        method=method_used,
        ci_level=ci_level,
        standard_error=reps.std(axis=0, ddof=1),
    )


def _bca_interval(
    data: Any, statistic: Callable, estimate: np.ndarray, reps: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Bias-corrected and accelerated interval endpoints (Efron 1987)."""
    n = _n_units(data)
    # bias correction z0 from the fraction of replicates below the point estimate
    prop = np.mean(reps < estimate, axis=0)
    prop = np.clip(prop, 1.0 / (reps.shape[0] + 1), 1.0 - 1.0 / (reps.shape[0] + 1))
    z0 = norm.ppf(prop)
    # acceleration from the jackknife skewness of the leave-one-out estimates
    jack = np.empty((n,) + estimate.shape, dtype=float)
    all_idx = np.arange(n)
    for i in range(n):
        jack[i] = _call(statistic, _take(data, np.delete(all_idx, i)))
    jack_mean = jack.mean(axis=0)
    diff = jack_mean - jack
    num = np.sum(diff**3, axis=0)
    den = 6.0 * (np.sum(diff**2, axis=0) ** 1.5)
    with np.errstate(invalid="ignore", divide="ignore"):
        accel = np.where(den != 0, num / den, 0.0)
    z_lo, z_hi = norm.ppf(alpha / 2.0), norm.ppf(1.0 - alpha / 2.0)

    def _adj(z: float) -> np.ndarray:
        zz = z0 + (z0 + z) / (1.0 - accel * (z0 + z))
        return norm.cdf(zz)

    a1 = _adj(z_lo)
    a2 = _adj(z_hi)
    lo = np.empty(estimate.shape) if estimate.ndim else np.empty(())
    hi = np.empty_like(lo)
    a1f, a2f, repsf = np.atleast_1d(a1), np.atleast_1d(a2), reps.reshape(reps.shape[0], -1)
    lo_flat = np.array([np.quantile(repsf[:, k], a1f[k]) for k in range(repsf.shape[1])])
    hi_flat = np.array([np.quantile(repsf[:, k], a2f[k]) for k in range(repsf.shape[1])])
    return lo_flat.reshape(estimate.shape), hi_flat.reshape(estimate.shape)


def block_bootstrap(
    data: Any,
    statistic: Callable[..., Any],
    block_length: int,
    *,
    n_boot: int = 2000,
    ci_level: float = 0.95,
    seed: int | RandomState | None = 0,
) -> BootstrapResult:
    """Moving-block bootstrap for serially dependent (time-series) data.

    Convenience wrapper over :func:`bootstrap` with ``block_length`` set: resamples contiguous blocks
    so within-block autocorrelation is preserved. Choose ``block_length`` on the order of the series'
    correlation length.
    """
    return bootstrap(
        data, statistic, n_boot=n_boot, method="percentile", ci_level=ci_level, seed=seed, block_length=block_length
    )


def wild_bootstrap(
    fitted: np.ndarray,
    residuals: np.ndarray,
    statistic: Callable[[np.ndarray], Any],
    *,
    n_boot: int = 2000,
    kind: str = "rademacher",
    ci_level: float = 0.95,
    seed: int | RandomState | None = 0,
) -> BootstrapResult:
    """Wild (residual-multiplier) bootstrap, robust to heteroscedasticity.

    Builds synthetic responses ``y* = fitted + residual * v`` where ``v`` are mean-zero,
    unit-variance two-point multipliers drawn independently per observation, then recomputes the
    statistic on each ``y*``. Because each residual keeps its own magnitude, the procedure preserves
    heteroscedasticity that an i.i.d. residual resample would destroy.

    Args:
        fitted: ``(n,)`` fitted values from the model.
        residuals: ``(n,)`` residuals ``y - fitted``.
        statistic: maps a synthetic response vector ``y*`` to a scalar or vector (e.g. refit and
            return coefficients).
        n_boot: number of resamples.
        kind: ``"rademacher"`` (``v in {-1, +1}``) or ``"mammen"`` (Mammen's two-point distribution).
        ci_level: central probability of the percentile interval.
        seed: RNG seed.

    Returns:
        A :class:`BootstrapResult` (percentile interval).
    """
    rng = _as_rng(seed)
    fitted = np.asarray(fitted, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    n = fitted.shape[0]
    # the observed response is y = fitted + residuals
    estimate = np.asarray(statistic(fitted + residuals), dtype=float)
    reps = np.empty((n_boot,) + estimate.shape, dtype=float)
    sqrt5 = np.sqrt(5.0)
    p_mammen = (sqrt5 + 1.0) / (2.0 * sqrt5)
    for b in range(n_boot):
        if kind == "rademacher":
            v = rng.choice([-1.0, 1.0], size=n)
        elif kind == "mammen":
            a = -(sqrt5 - 1.0) / 2.0
            c = (sqrt5 + 1.0) / 2.0
            v = np.where(rng.rand(n) < p_mammen, a, c)
        else:
            raise ValueError("kind must be 'rademacher' or 'mammen'.")
        reps[b] = np.asarray(statistic(fitted + residuals * v), dtype=float)
    alpha = 1.0 - ci_level
    lo = np.quantile(reps, alpha / 2.0, axis=0)
    hi = np.quantile(reps, 1.0 - alpha / 2.0, axis=0)
    return BootstrapResult(
        estimate=estimate,
        ci_low=np.asarray(lo),
        ci_high=np.asarray(hi),
        distribution=reps,
        method="wild-" + kind,
        ci_level=ci_level,
        standard_error=reps.std(axis=0, ddof=1),
    )


@dataclass
class PermutationResult:
    """Result of a :func:`permutation_test`.

    Attributes:
        statistic: the observed test statistic.
        pvalue: the (one- or two-sided) p-value.
        null_distribution: the statistic under each sampled/enumerated rearrangement.
        n_perm: number of rearrangements used.
        exact: True if the full permutation set was enumerated.
        alternative: the alternative hypothesis.
    """

    statistic: float
    pvalue: float
    null_distribution: np.ndarray
    n_perm: int
    exact: bool
    alternative: str


def _mean_diff(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(x) - np.mean(y))


def _pvalue(observed: float, null: np.ndarray, alternative: str, *, exact: bool = False) -> float:
    """Permutation p-value: ``count / n`` when the full group was enumerated, else Monte-Carlo.

    With ``exact=True`` the null set already contains the identity rearrangement, so ``count / n``
    IS the exact p-value; the ``(count + 1) / (n + 1)`` finite-sample correction (which adds the
    identity to a *random* sample of rearrangements) would double-count it.
    """
    n = null.size
    if alternative == "greater":
        count = np.sum(null >= observed)
    elif alternative == "less":
        count = np.sum(null <= observed)
    elif alternative == "two-sided":
        count = np.sum(np.abs(null) >= abs(observed))
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'.")
    if exact:
        return float(count / n)
    return float((count + 1) / (n + 1))


def permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    *,
    statistic: Callable[[np.ndarray, np.ndarray], float] | None = None,
    n_perm: int = 10000,
    alternative: str = "two-sided",
    paired: bool = False,
    stratify: np.ndarray | None = None,
    seed: int | RandomState | None = 0,
    exact_max: int = 10000,
) -> PermutationResult:
    """Two-sample permutation test for an arbitrary statistic under a sharp null.

    Under the null that the two samples are exchangeable, the labels can be shuffled freely; the
    statistic's permutation distribution is the reference. For ``two-sided`` the statistic is centered
    at zero by construction (difference statistics) and compared on absolute value.

    Args:
        x, y: the two samples (1-D). For ``paired=True`` they must have equal length and pairing is
            preserved by sign-flipping the within-pair differences.
        statistic: ``f(x, y) -> float``; defaults to the difference in means. For ``paired`` it is
            applied to ``(differences, zeros)`` so the default reduces to the mean paired difference.
        n_perm: number of random rearrangements (ignored if the exact set is enumerated).
        alternative: ``"two-sided"``, ``"greater"``, or ``"less"``.
        paired: paired (sign-flip) permutation instead of label shuffling.
        stratify: ``(n,)`` group labels (concatenated x-then-y) for **restricted** permutation --
            labels are shuffled only within each group, preserving group structure.
        seed: RNG seed.
        exact_max: if the number of distinct rearrangements is ``<= exact_max`` they are enumerated
            for an exact p-value.

    Returns:
        A :class:`PermutationResult`.
    """
    rng = _as_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    stat = statistic if statistic is not None else _mean_diff

    if paired:
        if x.shape != y.shape:
            raise ValueError("paired test needs x and y of equal length.")
        d = x - y
        observed = stat(d, np.zeros_like(d))
        n = d.shape[0]
        exact = 2**n <= exact_max
        if exact:
            null = np.empty(2**n)
            for k in range(2**n):
                signs = np.array([1.0 if (k >> j) & 1 else -1.0 for j in range(n)])
                null[k] = stat(d * signs, np.zeros_like(d))
        else:
            null = np.empty(n_perm)
            for p in range(n_perm):
                signs = rng.choice([-1.0, 1.0], size=n)
                null[p] = stat(d * signs, np.zeros_like(d))
        pval = _pvalue(observed, null, alternative, exact=exact)
        return PermutationResult(observed, pval, null, null.size, exact, alternative)

    observed = stat(x, y)
    pooled = np.concatenate([x, y])
    nx = x.shape[0]
    labels = np.concatenate([np.zeros(nx, dtype=int), np.ones(y.shape[0], dtype=int)])

    if stratify is not None:
        strata = np.asarray(stratify)
        null = np.empty(n_perm)
        for p in range(n_perm):
            perm = labels.copy()
            for g in np.unique(strata):
                pos = np.nonzero(strata == g)[0]
                perm[pos] = rng.permutation(perm[pos])
            null[p] = stat(pooled[perm == 0], pooled[perm == 1])
        return PermutationResult(observed, _pvalue(observed, null, alternative), null, n_perm, False, alternative)

    from math import comb

    n_total = pooled.shape[0]
    exact = comb(n_total, nx) <= exact_max
    if exact:
        idx_all = np.arange(n_total)
        combos = list(combinations(idx_all, nx))
        null = np.empty(len(combos))
        for i, c in enumerate(combos):
            mask = np.zeros(n_total, dtype=bool)
            mask[list(c)] = True
            null[i] = stat(pooled[mask], pooled[~mask])
    else:
        null = np.empty(n_perm)
        for p in range(n_perm):
            perm = rng.permutation(pooled)
            null[p] = stat(perm[:nx], perm[nx:])
    pval = _pvalue(observed, null, alternative, exact=exact)
    return PermutationResult(observed, pval, null, null.size, exact, alternative)


__all__ = [
    "BootstrapResult",
    "bootstrap",
    "block_bootstrap",
    "wild_bootstrap",
    "PermutationResult",
    "permutation_test",
]
