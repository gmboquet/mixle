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
    "Exact" always means exact FOR THE STATED SHARP NULL, carried on the result as
    ``null_hypothesis`` (STAT-RR17-12): the sign-flip null is SYMMETRY of the paired differences
    about the null value, not mean equality (a mean-zero asymmetric alternative rejected 43% at
    nominal 5%), and the two-sample null is full exchangeability, not equal means under unequal
    variances.

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


def _integer_control(name: str, value: int, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _ci_level(value: float) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        or not 0.0 < float(value) < 1.0
    ):
        raise ValueError("ci_level must be a finite number strictly between 0 and 1")
    return float(value)


def _validate_data(data: Any) -> int:
    parts = list(data) if _is_tuple(data) else [data]
    if not parts:
        raise ValueError("data must contain at least one array")
    lengths = []
    for part in parts:
        array = np.asarray(part)
        if array.ndim == 0:
            raise ValueError("each data part must have a non-empty observation axis")
        lengths.append(len(array))
    if not lengths[0] or any(length != lengths[0] for length in lengths):
        raise ValueError("all data parts must share the same non-empty first axis")
    return lengths[0]


def _labels(name: str, values: Any, n: int) -> np.ndarray:
    labels = np.asarray(values)
    if labels.ndim != 1 or len(labels) != n:
        raise ValueError(f"{name} must be one-dimensional with length {n}")
    for value in labels.tolist():
        if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
            raise ValueError(f"{name} cannot contain missing labels")
        try:
            hash(value)
        except TypeError as exc:
            raise ValueError(f"{name} must contain hashable scalar labels") from exc
    return labels


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
    result = np.asarray(out, dtype=float)
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError("statistic must return a non-empty finite scalar or fixed-shape array")
    return result


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
        # CIRCULAR moving blocks (audit RS-4): with non-circular starts in [0, n - l], observation
        # i appears in only min(i+1, l, n-i) blocks, so the endpoints are systematically
        # under-weighted and the resample mean is biased away from the sample mean by O(l/n).
        # Wrapping the series makes every observation appear in exactly l blocks, which removes
        # the centring bias exactly while preserving the same within-block dependence.
        n_blocks = int(np.ceil(n / block_length))
        starts = rng.randint(0, n, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_length) for s in starts]) % n
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
            ``"bca"``'s second-order accuracy holds for plain i.i.d. resampling of a SMOOTH
            (asymptotically normal, differentiable) functional; with ``groups`` / ``clusters`` /
            ``block_length`` / ``m`` set it falls back to ``"percentile"``. Non-smooth statistics
            -- sample quantiles and the median above all -- violate the smoothness the jackknife
            acceleration needs (the leave-one-out vector collapses to a couple of distinct
            values); when that degeneracy is detected the interval falls back to ``"percentile"``
            and ``method`` records ``"percentile (bca-degenerate-jackknife)"``.
        ci_level: central probability of the interval.
        seed: RNG seed.
        groups: ``(n,)`` labels for **stratified** resampling (resample within each group).
        clusters: ``(n,)`` labels for **cluster** resampling (resample whole clusters with
            replacement). Validity is asymptotic in the NUMBER OF CLUSTERS, not the number of
            rows: at G = 5 clusters, nominal-95% coverage measures ~0.5-0.7 however many rows
            each cluster holds, so treat few-cluster intervals as descriptive. Unequal cluster
            sizes additionally inflate replicate variance (each resample's row count varies with
            which clusters were drawn).
        block_length: moving-**block** length for serially dependent (time-series) data. Blocks
            are CIRCULAR (the series wraps), which removes the endpoint under-weighting of
            non-circular moving blocks. For picking the length, the coverage-optimal rate grows
            with the sample -- on the order of ``n**(1/3)`` scaled by the dependence strength --
            so "the correlation length" alone is materially too short for long series: a length
            that ignores ``n`` leaves residual dependence between blocks uncounted.
        m: subsample size for **m-out-of-n** subsampling (without replacement), ``m < n``.
            Replicates are rescaled about the point estimate by ``sqrt(m/n)`` (Politis--Romano),
            which assumes the usual ``sqrt(n)``-consistent statistic (a faster-rate statistic
            gets a conservative interval), and the interval takes the BASIC (pivotal)
            orientation that subsampling theory yields -- ``method`` is overridden in this mode.

    Returns:
        A :class:`BootstrapResult`.
    """
    if not callable(statistic):
        raise TypeError("statistic must be callable")
    if method not in ("percentile", "basic", "bca"):
        raise ValueError("method must be 'percentile', 'basic', or 'bca'.")
    n_boot = _integer_control("n_boot", n_boot, minimum=2)
    ci_level = _ci_level(ci_level)
    n = _validate_data(data)
    if block_length is not None:
        block_length = _integer_control("block_length", block_length, minimum=1)
        if block_length > n:
            raise ValueError("block_length must be in [1, n]")
    if m is not None:
        m = _integer_control("m", m, minimum=1)
        if m >= n:
            # m == n draws a without-replacement PERMUTATION of the full sample, so every
            # replicate of a permutation-invariant statistic equals the estimate exactly and the
            # returned "interval" has width zero -- a silent 0%-coverage confidence interval
            # (audit RS-9). Subsampling needs a genuinely smaller m.
            raise ValueError("m-out-of-n subsampling needs m < n; at m == n every replicate equals the estimate")
    if groups is not None:
        groups = _labels("groups", groups, n)
    if clusters is not None:
        clusters = _labels("clusters", clusters, n)
        if len(np.unique(clusters)) < 2:
            raise ValueError("cluster bootstrap requires at least two independent clusters")
    n_special = sum(x is not None for x in (groups, clusters, block_length, m))
    if n_special > 1:
        raise ValueError(
            "bootstrap() accepts at most one of groups, clusters, block_length, m; got "
            f"{n_special} set simultaneously -- _resample_indices would silently honor only one "
            "(clusters, then groups, then block_length, then m) and drop the rest."
        )
    rng = _as_rng(seed)
    estimate = _call(statistic, data)
    reps = np.empty((n_boot,) + estimate.shape, dtype=float)
    special = groups is not None or clusters is not None or block_length is not None or m is not None
    for b in range(n_boot):
        idx = _resample_indices(n, rng, groups=groups, clusters=clusters, block_length=block_length, m=m)
        reps[b] = _call(statistic, _take(data, idx))
    if m is not None:
        # Politis-Romano m-out-of-n rescaling: a size-m subsample statistic fluctuates at the
        # sqrt(m) rate, so shrink the replicates about the point estimate by sqrt(m/n) to put
        # them at full-sample scale (assumes the usual sqrt(n)-consistent statistic; a faster
        # rate needs a custom rescale and this interval is conservative for it).
        reps = estimate + np.sqrt(m / n) * (reps - estimate)

    alpha = 1.0 - ci_level
    if m is not None:
        # Subsampling theory approximates the law of (theta_hat_n - theta) by the law of
        # (theta_hat_m - theta_hat_n), which yields the BASIC (pivotal) orientation:
        # [2*est - Q_hi, 2*est - Q_lo]. The percentile orientation is its REFLECTION and agrees
        # only for symmetric limit laws -- subsampling exists for the asymmetric ones, and on
        # the canonical U(0, theta) sample-max example the percentile upper endpoint equals the
        # sample max, which sits below theta almost surely: coverage exactly zero (audit RS-1).
        method_used = "basic"
    elif method == "bca" and special:
        method_used = "percentile"
    else:
        method_used = method

    if method_used == "bca":
        bca = _bca_interval(data, statistic, estimate, reps, alpha)
        if bca is None:
            # Degenerate jackknife (audit RS-2): the leave-one-out vector of a quantile-type
            # statistic takes so few distinct values that the acceleration estimate is noise, not
            # skewness. Percentile is the honest fallback, and the label says why.
            method_used = "percentile (bca-degenerate-jackknife)"
        else:
            lo, hi = bca
    if method_used.startswith("percentile"):
        lo = np.quantile(reps, alpha / 2.0, axis=0)
        hi = np.quantile(reps, 1.0 - alpha / 2.0, axis=0)
    elif method_used == "basic":
        lo = 2.0 * estimate - np.quantile(reps, 1.0 - alpha / 2.0, axis=0)
        hi = 2.0 * estimate - np.quantile(reps, alpha / 2.0, axis=0)

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
) -> tuple[np.ndarray, np.ndarray] | None:
    """Bias-corrected and accelerated interval endpoints (Efron 1987).

    Returns ``None`` when the jackknife is degenerate (fewer than 3 distinct leave-one-out values
    for some component) -- the acceleration is a skewness estimate and needs actual variation to
    estimate it; the caller then falls back to the percentile interval with a labelled method.
    """
    n = _n_units(data)
    if n < 2:
        # the jackknife loop below leaves one observation out per iteration; at n=1 that computes
        # the statistic on an EMPTY sample (NaN, usually with its own RuntimeWarning), and since
        # `nan != 0` is True in numpy the `den != 0` guard a few lines down does not catch it either
        # -- the NaN silently propagated through norm.cdf/np.quantile and surfaced downstream as an
        # unrelated "Quantiles must be in the range [0, 1]" ValueError instead of this clear one.
        raise ValueError(f"BCa interval needs at least 2 observations for the jackknife acceleration term, got {n}.")
    # Bias correction z0 with the MID-P convention (audit RS-3): a lattice statistic puts a point
    # mass exactly AT the estimate, and counting none of it (strict <) reads the atom as downward
    # bias -- an unbiased symmetric discrete statistic got a spurious correction. Counting half
    # the atom makes z0 = 0 exactly in that symmetric case and changes nothing for continuous
    # statistics (where ties have probability zero).
    prop = np.mean(reps < estimate, axis=0) + 0.5 * np.mean(reps == estimate, axis=0)
    prop = np.clip(prop, 1.0 / (reps.shape[0] + 1), 1.0 - 1.0 / (reps.shape[0] + 1))
    z0 = norm.ppf(prop)
    # acceleration from the jackknife skewness of the leave-one-out estimates
    jack = np.empty((n,) + estimate.shape, dtype=float)
    all_idx = np.arange(n)
    for i in range(n):
        jack[i] = _call(statistic, _take(data, np.delete(all_idx, i)))
    # Degenerate jackknife detection (audit RS-2): the acceleration is consistent for smooth
    # functionals; for sample quantiles the leave-one-out vector collapses to ~2 distinct values
    # and the ratio below estimates noise. Bail out to percentile rather than decorate the
    # interval with a meaningless skewness adjustment.
    jack_flat = jack.reshape(n, -1)
    if any(np.unique(jack_flat[:, k]).size < 3 for k in range(jack_flat.shape[1])):
        return None
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
    leverage: np.ndarray | None = None,
) -> BootstrapResult:
    """Wild (residual-multiplier) bootstrap for heteroscedastic regression errors.

    Builds synthetic responses ``y* = fitted + residual * v`` where ``v`` are mean-zero,
    unit-variance two-point multipliers drawn independently per observation, then recomputes the
    statistic on each ``y*``. Because each residual keeps its own magnitude, the procedure preserves
    heteroscedasticity that an i.i.d. residual resample would destroy.

    The robustness claim is conditional on what the residuals carry. OLS residuals systematically
    UNDERSTATE the error at influential design points (``E[e_i^2] = sigma_i^2 (1 - h_i)`` with
    ``h_i`` the hat-matrix diagonal), so multiplying raw residuals hands high-leverage
    observations too little noise exactly where mis-estimated variance hurts the most. Pass
    ``leverage`` (the hat diagonals) to apply the Davidson-Flachaire restoration
    ``e_i / (1 - h_i)`` before multiplication -- the standard wild-bootstrap practice for
    regression coefficients. ``residuals`` must be RESPONSE-SCALE ``y - fitted``: Pearson,
    deviance, or standardized residuals reconstruct a different response vector and silently
    change the estimand (the point estimate is computed from ``fitted + residuals``, which equals
    the observed ``y`` only for raw residuals).

    Args:
        fitted: ``(n,)`` fitted values from the model.
        residuals: ``(n,)`` raw response-scale residuals ``y - fitted``.
        statistic: maps a synthetic response vector ``y*`` to a scalar or vector (e.g. refit and
            return coefficients).
        n_boot: number of resamples.
        kind: ``"rademacher"`` (``v in {-1, +1}``) or ``"mammen"`` (Mammen's two-point distribution).
        ci_level: central probability of the percentile interval.
        seed: RNG seed.
        leverage: optional ``(n,)`` hat-matrix diagonals ``h_i in [0, 1)``; when given, residuals
            are inflated to ``e_i / (1 - h_i)`` inside the resampling loop (the point estimate
            still uses the raw residuals, i.e. the observed response).

    Returns:
        A :class:`BootstrapResult` (percentile interval).
    """
    if not callable(statistic):
        raise TypeError("statistic must be callable")
    if kind not in ("rademacher", "mammen"):
        raise ValueError("kind must be 'rademacher' or 'mammen'.")
    n_boot = _integer_control("n_boot", n_boot, minimum=2)
    ci_level = _ci_level(ci_level)
    rng = _as_rng(seed)
    fitted = np.asarray(fitted, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    if (
        fitted.ndim != 1
        or residuals.ndim != 1
        or fitted.size == 0
        or fitted.shape != residuals.shape
        or not np.all(np.isfinite(fitted))
        or not np.all(np.isfinite(residuals))
    ):
        raise ValueError("fitted and residuals must be matching non-empty finite vectors")
    n = fitted.shape[0]
    if leverage is not None:
        leverage = np.asarray(leverage, dtype=float)
        if leverage.shape != fitted.shape or not np.all(np.isfinite(leverage)):
            raise ValueError("leverage must be a finite vector aligned with fitted")
        if np.any((leverage < 0.0) | (leverage >= 1.0)):
            raise ValueError("leverage entries must be hat diagonals in [0, 1)")
        loop_residuals = residuals / (1.0 - leverage)
    else:
        loop_residuals = residuals
    # the observed response is y = fitted + residuals (raw residuals by contract)
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
        reps[b] = _call(statistic, fitted + loop_residuals * v)
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
    # STAT-RR17-12: the sharp null the rearrangement group actually tests, carried WITH the
    # p-value. "exact" means exact FOR THIS NULL -- the paired sign-flip null is symmetry of the
    # differences about null_value, and for mean-zero ASYMMETRIC differences the enumerated
    # "exact" test rejected 43% at nominal 5% (n=8, +9 w.p. 0.1 else -1): it is not a
    # mean-equality test. The two-sample null is exchangeability (F = G); with unequal variances
    # and sizes the raw mean-difference statistic does not control the equal-means level
    # (studentize for that).
    null_hypothesis: str = ""


def _mean_diff(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(x) - np.mean(y))


def _pvalue(
    observed: float,
    null: np.ndarray,
    alternative: str,
    *,
    exact: bool = False,
    null_value: float = 0.0,
) -> float:
    """Permutation p-value: ``count / n`` when the full group was enumerated, else Monte-Carlo.

    With ``exact=True`` the null set already contains the identity rearrangement, so ``count / n``
    IS the exact p-value; the ``(count + 1) / (n + 1)`` finite-sample correction (which adds the
    identity to a *random* sample of rearrangements) would double-count it.
    """
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'.")
    null = np.asarray(null, dtype=float)
    if null.ndim != 1 or null.size == 0 or not np.all(np.isfinite(null)) or not np.isfinite(observed):
        raise ValueError("permutation inference requires a non-empty finite null distribution and statistic")
    if not np.isfinite(null_value):
        raise ValueError("null_value must be finite")
    n = null.size
    if alternative == "greater":
        count = np.sum(null >= observed)
    elif alternative == "less":
        count = np.sum(null <= observed)
    elif alternative == "two-sided":
        count = np.sum(np.abs(null - null_value) >= abs(observed - null_value))
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
    null_value: float | None = None,
) -> PermutationResult:
    """Two-sample permutation test for an arbitrary statistic under a sharp null.

    Under the null that the two samples are exchangeable, the labels can be shuffled freely; the
    statistic's permutation distribution is the reference. For ``two-sided`` the statistic is centered
    at zero by construction (difference statistics) and compared on absolute value.

    Args:
        x, y: the two samples (1-D), each with at least one observation. For ``paired=True`` they
            must have equal length and pairing is preserved by sign-flipping the within-pair
            differences.
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
        null_value: reference value for a custom two-sided statistic. Required when ``statistic`` is
            supplied with ``alternative="two-sided"`` so the absolute tail is centered correctly.

    Returns:
        A :class:`PermutationResult`.

    Raises:
        ValueError: if ``x`` or ``y`` is empty. A single observation per group (or a single paired
            difference) is still a legitimate, if weak, test -- e.g. one-vs-one has 2 distinct label
            swaps and correctly returns ``pvalue=1.0``, not an error. But an empty group has ZERO
            rearrangements to enumerate: exact enumeration falls back to ``comb(n, 0) == 1``, silently
            "enumerating" one degenerate combination that recomputes ``stat`` on the same empty-vs-
            everything split, producing a null distribution of a single NaN. ``NaN >= NaN`` is False
            in numpy, so the observed-vs-null comparison in :func:`_pvalue` counts 0 exceedances out
            of 1 and returns an exact p-value of ``0.0`` -- reading as maximal significance from zero
            evidence instead of failing loudly.
    """
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'.")
    n_perm = _integer_control("n_perm", n_perm, minimum=1)
    exact_max = _integer_control("exact_max", exact_max, minimum=1)
    if statistic is not None and not callable(statistic):
        raise TypeError("statistic must be callable")
    if alternative == "two-sided" and statistic is not None and null_value is None:
        raise ValueError("a custom two-sided statistic requires an explicit finite null_value")
    null_center = 0.0 if null_value is None else float(null_value)
    if not np.isfinite(null_center):
        raise ValueError("null_value must be finite")
    if paired and stratify is not None:
        raise ValueError("stratify is not supported with paired sign-flip permutations")
    rng = _as_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x and y must be finite one-dimensional samples")
    if x.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError(
            f"permutation_test requires at least one observation in each of x and y, got {x.shape[0]} and {y.shape[0]}."
        )
    stat = statistic if statistic is not None else _mean_diff

    if paired:
        if x.shape != y.shape:
            raise ValueError("paired test needs x and y of equal length.")
        d = x - y
        observed = float(stat(d, np.zeros_like(d)))
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
        pval = _pvalue(observed, null, alternative, exact=exact, null_value=null_center)
        return PermutationResult(
            observed,
            pval,
            null,
            null.size,
            exact,
            alternative,
            null_hypothesis=(
                "paired differences are SYMMETRIC about the null value (sign-flip "
                "exchangeability); exact for THAT null only -- for mean-zero asymmetric "
                "differences the enumerated test rejected 43% at nominal 5% (n=8), so this is "
                "not a mean-equality test under asymmetry"
            ),
        )

    observed = float(stat(x, y))
    pooled = np.concatenate([x, y])
    nx = x.shape[0]
    labels = np.concatenate([np.zeros(nx, dtype=int), np.ones(y.shape[0], dtype=int)])

    if stratify is not None:
        strata = _labels("stratify", stratify, len(pooled))
        swappable = False
        for group in np.unique(strata):
            group_labels = labels[strata == group]
            swappable = swappable or (np.any(group_labels == 0) and np.any(group_labels == 1))
        if not swappable:
            raise ValueError("stratified permutation requires at least one stratum containing both samples")
        null = np.empty(n_perm)
        for p in range(n_perm):
            perm = labels.copy()
            for g in np.unique(strata):
                pos = np.nonzero(strata == g)[0]
                perm[pos] = rng.permutation(perm[pos])
            null[p] = stat(pooled[perm == 0], pooled[perm == 1])
        return PermutationResult(
            observed,
            _pvalue(observed, null, alternative, null_value=null_center),
            null,
            n_perm,
            False,
            alternative,
            null_hypothesis="within-stratum exchangeability of the two samples (F = G per stratum)",
        )

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
    pval = _pvalue(observed, null, alternative, exact=exact, null_value=null_center)
    return PermutationResult(
        observed,
        pval,
        null,
        null.size,
        exact,
        alternative,
        null_hypothesis=(
            "exchangeability of the two samples (F = G); with unequal variances and sizes the "
            "raw mean-difference statistic does not control an equal-MEANS level -- studentize "
            "the statistic for that estimand"
        ),
    )


__all__ = [
    "BootstrapResult",
    "bootstrap",
    "block_bootstrap",
    "wild_bootstrap",
    "PermutationResult",
    "permutation_test",
]
