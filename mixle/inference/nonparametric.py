"""Classical nonparametric (rank-based) hypothesis tests.

Distribution-free two-sample, k-sample, paired, repeated-measures, ordered-alternative, and
goodness-of-fit tests, each returning a small result object with the statistic, p-value, and -- where
standard -- an effect size. Statistics are computed here (mid-ranks for ties); tail probabilities use
exact small-sample nulls where SciPy / R use them (the Wilcoxon signed-rank enumeration at
``n <= 25`` without ties or zeros; the two-sample KS at small samples) and the asymptotic reference
distributions (normal / chi-square / Student-t / Kolmogorov) with tie corrections otherwise. Not
every asymptotic branch carries a continuity correction (the Page test does not; the runs test is
EXACT at n <= 60), and the remaining small-sample normal approximations are approximations -- see
each test's docstring.

.. parsed-literal::

  Two independent samples : :func:`mann_whitney_u` (Wilcoxon rank-sum), :func:`brunner_munzel`,
                            :func:`cliffs_delta`, :func:`ks_2samp`
  k independent samples   : :func:`kruskal_wallis`, :func:`mood_median_test`, :func:`dunn_test` (post-hoc)
  Paired / one sample     : :func:`wilcoxon_signed_rank`, :func:`sign_test`
  Repeated measures       : :func:`friedman_test`
  Ordered alternatives    : :func:`jonckheere_terpstra` (independent), :func:`page_trend_test` (repeated)
  Goodness of fit / 1-samp: :func:`ks_1samp`, :func:`runs_test`
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats


def _alternative(value: str, allowed: tuple[str, ...] = ("two-sided", "greater", "less")) -> str:
    if value not in allowed:
        raise ValueError(f"alternative must be one of {allowed}")
    return value


def _sample(name: str, values: Any, *, minimum: int = 1) -> np.ndarray:
    try:
        sample = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite one-dimensional sample") from exc
    if sample.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if len(sample) < minimum:
        raise ValueError(f"{name} must contain at least {minimum} observations")
    if not np.all(np.isfinite(sample)):
        raise ValueError(f"{name} must contain only finite observations")
    return sample


def _groups(samples: tuple[Any, ...], *, minimum_groups: int = 2, minimum_size: int = 1) -> list[np.ndarray]:
    if len(samples) < minimum_groups:
        raise ValueError(f"test requires at least {minimum_groups} samples")
    return [_sample(f"sample {index}", values, minimum=minimum_size) for index, values in enumerate(samples)]


def _related(measurements: tuple[Any, ...], *, minimum_groups: int = 3) -> np.ndarray:
    groups = _groups(measurements, minimum_groups=minimum_groups, minimum_size=2)
    if any(len(group) != len(groups[0]) for group in groups[1:]):
        raise ValueError("related samples must have matching lengths")
    return np.column_stack(groups)


def _ranks(a: np.ndarray) -> np.ndarray:
    return stats.rankdata(a)


def _tie_term(a: np.ndarray) -> float:
    """``sum(t**3 - t)`` over tie-group sizes -- the standard rank-variance tie correction."""
    _, counts = np.unique(a, return_counts=True)
    return float(np.sum(counts**3 - counts))


# --- two independent samples ------------------------------------------------
@dataclass
class MannWhitneyResult:
    """Result of a two-sample Mann-Whitney U test."""

    statistic: float  # the U statistic for the first sample
    statistic2: float  # the U statistic for the second sample (= n1*n2 - statistic)
    zscore: float
    pvalue: float
    rank_biserial: float  # effect size in [-1, 1]
    alternative: str
    # STAT-RR17-17: the variance is an exchangeability (F = G) variance; testing stochastic
    # ordering under unequal shapes with it inflates the level (12.5% measured at nominal 5%)
    null_hypothesis: str = "exchangeability (F = G); use brunner_munzel for unequal shapes"


def mann_whitney_u(x: Any, y: Any, *, alternative: str = "two-sided", use_continuity: bool = True) -> MannWhitneyResult:
    """Mann-Whitney U / Wilcoxon rank-sum test for two independent samples.

    THE NULL IS FULL EXCHANGEABILITY (``F = G``), not mere stochastic equality: the reference
    variance is derived under identical distributions, so with equal ``P(X > Y) = 1/2`` but
    UNEQUAL shapes/spreads and unbalanced sample sizes the level is not controlled -- measured
    12.5% rejection at nominal 5% one way and 1.5% with the sizes reversed (STAT-RR17-17). For
    the stochastic-equality null under unequal shapes use :func:`brunner_munzel`, whose variance
    estimates each group separately. Uses mid-ranks for ties, the tie-corrected normal
    approximation, and (default) a continuity correction. ``alternative`` is ``'two-sided'``,
    ``'greater'`` (x > y), or ``'less'``. The rank-biserial correlation ``2*U1/(n1 n2) - 1`` is
    reported as the effect size; ``null_hypothesis`` on the result names the null in force.
    """
    _alternative(alternative)
    x = _sample("x", x)
    y = _sample("y", y)
    n1, n2 = x.size, y.size
    pooled = np.concatenate([x, y])
    ranks = _ranks(pooled)
    r1 = float(ranks[:n1].sum())
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    n = n1 + n2
    mu = n1 * n2 / 2.0
    sigma = np.sqrt((n1 * n2 / 12.0) * ((n + 1) - _tie_term(pooled) / (n * (n - 1))))
    if sigma == 0:
        raise ValueError("Mann-Whitney test is undefined when all pooled ranks are tied")
    else:
        d = u1 - mu
        if use_continuity:  # shrink the gap toward the mean by 1/2
            d -= np.sign(d) * 0.5 if alternative == "two-sided" else 0.5 * (1 if d > 0 else -1)
        z = d / sigma
        if alternative == "two-sided":
            p = 2.0 * stats.norm.sf(abs(z))
        elif alternative == "greater":
            cc = 0.5 if use_continuity else 0.0
            z = (u1 - mu - cc) / sigma
            p = stats.norm.sf(z)
        elif alternative == "less":
            cc = 0.5 if use_continuity else 0.0
            z = (u1 - mu + cc) / sigma
            p = stats.norm.cdf(z)
        else:
            raise ValueError("alternative must be 'two-sided', 'greater', or 'less'.")
    rbc = 2.0 * u1 / (n1 * n2) - 1.0
    return MannWhitneyResult(float(u1), float(u2), float(z), float(min(p, 1.0)), float(rbc), alternative)


def cliffs_delta(x: Any, y: Any) -> float:
    """Cliff's delta effect size in [-1, 1]: ``P(x > y) - P(x < y)`` (rank-based, ties count as 0)."""
    x = _sample("x", x)
    y = _sample("y", y)
    diff = np.sign(x[:, None] - y[None, :])
    return float(diff.mean())


@dataclass
class TestResult:
    """Generic statistic + p-value result; ``extra`` carries test-specific fields (effect size, df, ...)."""

    statistic: float
    pvalue: float
    extra: dict[str, Any] = field(default_factory=dict)


def brunner_munzel(x: Any, y: Any, *, alternative: str = "two-sided", distribution: str = "t") -> TestResult:
    """Brunner-Munzel test: the generalized Wilcoxon test that does not assume equal variances/shapes.

    Tests the stochastic-equality null ``P(x < y) + 0.5 P(x = y) = 1/2``. ``alternative`` is
    ``'two-sided'``, ``'greater'`` (x > y), or ``'less'`` -- the same direction convention as
    :func:`mann_whitney_u` (and scipy). ``distribution='t'`` uses a Satterthwaite t reference
    (recommended for small samples); ``'normal'`` the normal approximation. Reports the estimated
    relative effect ``p_hat = P(x < y) + 0.5 P(x = y)`` in ``extra``.
    """
    _alternative(alternative)
    if distribution not in ("t", "normal"):
        raise ValueError("distribution must be 't' or 'normal'")
    x = _sample("x", x, minimum=2)
    y = _sample("y", y, minimum=2)
    n1, n2 = x.size, y.size
    rank_all = _ranks(np.concatenate([x, y]))
    rx, ry = _ranks(x), _ranks(y)
    r1m, r2m = rank_all[:n1].mean(), rank_all[n1:].mean()
    s1 = np.sum((rank_all[:n1] - rx - r1m + (n1 + 1) / 2.0) ** 2) / (n1 - 1)
    s2 = np.sum((rank_all[n1:] - ry - r2m + (n2 + 1) / 2.0) ** 2) / (n2 - 1)
    denom = n1 * s1 + n2 * s2
    if denom <= 0:
        raise ValueError("Brunner-Munzel test requires non-zero rank variation")
    w = n1 * n2 * (r2m - r1m) / ((n1 + n2) * np.sqrt(denom))
    p_hat = (r2m - (n2 + 1) / 2.0) / n1  # P(x < y) + 0.5 P(x = y)
    if distribution == "t":
        df_num = denom**2
        df_den = (n1 * s1) ** 2 / (n1 - 1) + (n2 * s2) ** 2 / (n2 - 1)
        df = df_num / df_den if df_den > 0 else np.inf
        dist = stats.t(df)
        extra = {"p_hat": float(p_hat), "df": float(df)}
    else:
        dist = stats.norm
        extra = {"p_hat": float(p_hat)}
    if alternative == "two-sided":
        p = 2.0 * dist.sf(abs(w))
    elif alternative == "greater":  # x > y pushes the y-ranks (and w) DOWN: lower tail
        p = dist.cdf(w)
    elif alternative == "less":
        p = dist.sf(w)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'.")
    return TestResult(float(w), float(min(p, 1.0)), extra)


def ks_2samp(x: Any, y: Any, *, alternative: str = "two-sided") -> TestResult:
    """Two-sample Kolmogorov-Smirnov test: max gap between the two empirical CDFs.

    The p-value is exact at small samples and asymptotic otherwise (scipy ``method='auto'``).
    The null distribution is distribution-free for CONTINUOUS data; with ties (discrete or
    rounded data) the p-value is conservative.
    """
    _alternative(alternative)
    x = np.sort(_sample("x", x))
    y = np.sort(_sample("y", y))
    n1, n2 = x.size, y.size
    allv = np.concatenate([x, y])
    cdf1 = np.searchsorted(x, allv, side="right") / n1
    cdf2 = np.searchsorted(y, allv, side="right") / n2
    diff = cdf1 - cdf2
    if alternative == "two-sided":
        d = float(np.max(np.abs(diff)))
    elif alternative == "greater":
        d = float(np.max(diff))
    elif alternative == "less":
        d = float(-np.min(diff))
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'.")
    # The p-value comes from scipy's two-sample machinery (method='auto': EXACT at small samples,
    # asymptotic otherwise). The previous hand-rolled version evaluated the ONE-sample kstwo law at
    # round(n1*n2/(n1+n2)) -- a heuristic substitution, not the two-sample null: at n1=n2=3 with
    # complete separation it returned p = 0.0 where the exact permutation p-value is 2/C(6,3) = 0.1
    # (audit NP-3). The statistic conventions match scipy's exactly, verified above.
    p = float(stats.ks_2samp(x, y, alternative=alternative, method="auto").pvalue)
    return TestResult(d, float(min(max(p, 0.0), 1.0)), {"n1": n1, "n2": n2})


def ks_1samp(x: Any, cdf: Callable[[np.ndarray], np.ndarray], *, alternative: str = "two-sided") -> TestResult:
    """One-sample Kolmogorov-Smirnov goodness-of-fit test against a fully-specified ``cdf`` callable."""
    _alternative(alternative)
    if not callable(cdf):
        raise TypeError("cdf must be callable")
    x = np.sort(_sample("x", x))
    n = x.size
    cdfv = np.asarray(cdf(x), dtype=float)
    if cdfv.shape != x.shape or not np.all(np.isfinite(cdfv)):
        raise ValueError("cdf must return one finite value per observation")
    if np.any((cdfv < 0.0) | (cdfv > 1.0)) or np.any(np.diff(cdfv) < 0.0):
        raise ValueError("cdf values must lie in [0, 1] and be non-decreasing")
    d_plus = float(np.max(np.arange(1, n + 1) / n - cdfv))
    d_minus = float(np.max(cdfv - np.arange(0, n) / n))
    if alternative == "two-sided":
        d = max(d_plus, d_minus)
        p = float(stats.kstwobign.sf(np.sqrt(n) * d))  # limiting KS distribution (matches scipy 'asymp')
    elif alternative == "greater":
        d = d_plus
        p = float(np.exp(-2.0 * n * d * d))
    elif alternative == "less":
        d = d_minus
        p = float(np.exp(-2.0 * n * d * d))
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'.")
    return TestResult(d, float(min(max(p, 0.0), 1.0)))


# --- k independent samples --------------------------------------------------
def kruskal_wallis(*samples: Any) -> TestResult:
    """Kruskal-Wallis H test: the rank-based k-sample generalization of Mann-Whitney (one-way ANOVA).

    Tie-corrected H with a chi-square(k-1) reference. ``extra`` carries ``df`` and the ``epsilon_squared``
    effect size ``(H - k + 1)/(N - k)``.
    """
    groups = _groups(samples)
    sizes = [g.size for g in groups]
    pooled = np.concatenate(groups)
    n = pooled.size
    if n <= len(groups) or np.unique(pooled).size < 2:
        raise ValueError("Kruskal-Wallis requires residual degrees of freedom and non-tied ranks")
    ranks = _ranks(pooled)
    idx, h_sum = 0, 0.0
    for sz in sizes:
        rsum = ranks[idx : idx + sz].sum()
        h_sum += rsum * rsum / sz
        idx += sz
    h = 12.0 / (n * (n + 1)) * h_sum - 3.0 * (n + 1)
    h /= 1.0 - _tie_term(pooled) / (n**3 - n)  # tie correction
    k = len(groups)
    df = k - 1
    p = float(stats.chi2.sf(h, df))
    eps2 = (h - k + 1) / (n - k)
    return TestResult(float(h), p, {"df": df, "epsilon_squared": float(eps2)})


def mood_median_test(*samples: Any, ties: str = "below") -> TestResult:
    """Mood's median test: chi-square test that k samples share a common median.

    Cross-tabulates each observation as above / (at-or-below) the pooled grand median and runs a
    chi-square test of independence on the resulting 2xk table. ``extra`` carries the ``grand_median``.
    """
    if ties not in ("below", "above"):
        raise ValueError("ties must be 'below' or 'above'")
    groups = _groups(samples)
    pooled = np.concatenate(groups)
    gm = float(np.median(pooled))
    above = [int(np.sum(g > gm)) for g in groups]
    if ties == "below":
        below = [g.size - a for g, a in zip(groups, above)]
    else:  # 'above' counts ties as above
        above = [int(np.sum(g >= gm)) for g in groups]
        below = [g.size - a for g, a in zip(groups, above)]
    table = np.array([above, below], dtype=float)
    if np.any(table.sum(axis=1) == 0) or np.any(table.sum(axis=0) == 0):
        raise ValueError("Mood median test requires observations on both sides of the pooled median")
    chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
    return TestResult(float(chi2), float(p), {"df": int(dof), "grand_median": gm})


@dataclass
class DunnResult:
    """Post-hoc Dunn pairwise comparisons after Kruskal-Wallis."""

    comparisons: list[tuple[int, int]]
    zscores: np.ndarray
    pvalues: np.ndarray  # adjusted
    p_adjust: str


def dunn_test(*samples: Any, p_adjust: str = "holm") -> DunnResult:
    """Dunn's post-hoc test: all pairwise rank-mean comparisons after a Kruskal-Wallis rejection.

    Uses the pooled-rank z statistic with the shared tie-corrected variance, and adjusts the pairwise
    p-values by ``'holm'``, ``'bonferroni'``, or ``'none'``.
    """
    if p_adjust not in ("holm", "bonferroni", "none"):
        raise ValueError("p_adjust must be 'holm', 'bonferroni', or 'none'.")
    groups = _groups(samples)
    sizes = [g.size for g in groups]
    pooled = np.concatenate(groups)
    n = pooled.size
    ranks = _ranks(pooled)
    means, idx = [], 0
    for sz in sizes:
        means.append(ranks[idx : idx + sz].mean())
        idx += sz
    tie = _tie_term(pooled)
    sigma2_base = (n * (n + 1) - tie / (n - 1)) / 12.0
    if sigma2_base <= 0:
        raise ValueError("Dunn test requires non-zero pooled rank variation")
    comps, zs, raw = [], [], []
    k = len(groups)
    for i in range(k):
        for j in range(i + 1, k):
            se = np.sqrt(sigma2_base * (1.0 / sizes[i] + 1.0 / sizes[j]))
            z = (means[i] - means[j]) / se if se > 0 else 0.0
            comps.append((i, j))
            zs.append(float(z))
            raw.append(2.0 * stats.norm.sf(abs(z)))
    raw = np.asarray(raw)
    m = raw.size
    if p_adjust == "bonferroni":
        adj = np.minimum(raw * m, 1.0)
    elif p_adjust == "holm":
        order = np.argsort(raw)
        adj = np.empty(m)
        running = 0.0
        for rank, k_ in enumerate(order):
            running = max(running, (m - rank) * raw[k_])
            adj[k_] = min(running, 1.0)
    elif p_adjust == "none":
        adj = raw
    return DunnResult(comps, np.asarray(zs), adj, p_adjust)


# --- paired / one sample ----------------------------------------------------
@dataclass
class WilcoxonResult:
    """Result of a paired or one-sample Wilcoxon signed-rank test.

    ``method`` names the null actually used (STAT-RR17-15): ``"exact"`` (full enumeration, the
    n <= 25 no-ties/no-zeros regime) or ``"normal"`` (tie/zero-corrected approximation).
    ``zscore`` is signed FOR THE STATED ALTERNATIVE -- positive means the data lean toward it --
    so an all-positive sample under ``alternative='greater'`` reports a positive z next to its
    small exact p, where the old min(W+, W-)-based z reported -2.02 beside p = 0.03125 and a
    rank-biserial of +1.
    """

    statistic: float  # the smaller of W+ / W- (test statistic)
    zscore: float
    pvalue: float
    rank_biserial: float
    alternative: str
    method: str = "normal"


def wilcoxon_signed_rank(
    x: Any, y: Any = None, *, alternative: str = "two-sided", zero_method: str = "wilcox", correction: bool = False
) -> WilcoxonResult:
    """Wilcoxon signed-rank test for paired samples (or one sample vs 0).

    Ranks ``|d|`` for ``d = x - y`` (mid-ranks for ties), splits into positive / negative rank sums,
    and uses the EXACT enumeration null when it is available -- ``n <= 25`` with no zero differences
    and no tied ``|d|`` -- and the tie-corrected normal approximation otherwise (the same regime
    switch SciPy and R make; the normal approximation is level-violating at very small ``n``, where
    its smallest attainable p sits below the exact one). ``zero_method='wilcox'`` drops zero
    differences (and their ranks); ``'pratt'`` keeps them in the ranking but drops them from the
    sums, with the matching Pratt/Cureton zero corrections applied to the null mean and variance
    (as scipy does). ``correction`` applies only to the normal branch (exact tails need no
    continuity repair). The matched-pairs rank-biserial correlation is reported as the effect size.
    """
    _alternative(alternative)
    if zero_method not in ("wilcox", "pratt"):
        raise ValueError("zero_method must be 'wilcox' or 'pratt'")
    x = _sample("x", x)
    if y is None:
        d = x
    else:
        y_array = _sample("y", y)
        if y_array.shape != x.shape:
            raise ValueError("paired samples must have matching shapes")
        d = x - y_array
    if zero_method == "wilcox":
        d = d[d != 0]
    n = d.size
    if n == 0:
        raise ValueError("Wilcoxon signed-rank test requires at least one non-zero difference")
    r = _ranks(np.abs(d))
    n_zero = 0
    if zero_method == "pratt":
        n_zero = int(np.sum(d == 0))
        keep = d != 0
        r, d = r[keep], d[keep]
        if d.size == 0:
            raise ValueError("Wilcoxon signed-rank test requires at least one non-zero difference")
    r_plus = float(r[d > 0].sum())
    r_minus = float(r[d < 0].sum())
    nn = d.size + n_zero  # the ranked count (zeros stay in the ranking under 'pratt')
    t = min(r_plus, r_minus)
    # Pratt/Cureton (1967) zero corrections: the zero block occupies the lowest ranks but contributes
    # to neither sum, so its share is subtracted from the null mean and variance (no-op when n_zero=0);
    # ties among the remaining |d| correct the variance as usual.
    mu = (nn * (nn + 1) - n_zero * (n_zero + 1)) / 4.0
    sigma = np.sqrt(
        (nn * (nn + 1) * (2 * nn + 1) - n_zero * (n_zero + 1) * (2 * n_zero + 1) - 0.5 * _tie_term(r)) / 24.0
    )
    if sigma == 0:
        raise ValueError("Wilcoxon signed-rank reference variance is zero")
    if nn <= 25 and n_zero == 0 and _tie_term(r) == 0.0:
        # EXACT null (audit NP-1): with no zeros and no ties the ranks are exactly {1..nn} and
        # T+ is a subset sum, enumerated by the 0/1 convolution prod_k (1 + x^k) -- 2^25 patterns
        # count exactly in float64. The normal approximation is not merely imprecise here, it is
        # level-violating: at n=5 the most extreme outcome gets normal p = 0.043 (< 0.05) while
        # the exact two-sided p is 2/32 = 0.0625, so the realized type-I rate at nominal 5% was
        # a guaranteed 6.25% (same construction at n=6). SciPy and R both switch to the exact
        # null in this regime, which is what the module-level conventions sentence promises.
        # The continuity correction is an approximation repair and does not apply to exact tails.
        counts = np.zeros(nn * (nn + 1) // 2 + 1)
        counts[0] = 1.0
        for k in range(1, nn + 1):
            counts[k:] = counts[k:] + counts[:-k].copy()
        cdf = np.cumsum(counts) / counts.sum()
        method = "exact"
        # descriptive z, SIGNED FOR THE ALTERNATIVE (STAT-RR17-15): r_plus above its null mean
        # leans toward 'greater'; two-sided keeps the magnitude of the departure with the sign
        # of (r_plus - mu) so direction and effect size read consistently
        z = ((r_plus - mu) / sigma) if alternative != "less" else ((mu - r_plus) / sigma)
        if alternative == "two-sided":
            z = (r_plus - mu) / sigma
        if alternative == "two-sided":
            p = min(1.0, 2.0 * float(cdf[int(t)]))
        elif alternative == "greater":  # x > y -> R+ large; P(T+ >= r_plus) = P(T+ <= r_minus)
            p = float(cdf[int(r_minus)])
        elif alternative == "less":
            p = float(cdf[int(r_plus)])
        else:
            raise ValueError("alternative must be 'two-sided', 'greater', or 'less'.")
    else:
        method = "normal"
        if alternative == "two-sided":
            cc = 0.5 if correction else 0.0
            z = (t - mu + cc) / sigma
            p = 2.0 * stats.norm.cdf(z)
        elif alternative == "greater":  # x > y -> R+ large
            cc = 0.5 if correction else 0.0
            z = (r_plus - mu - cc) / sigma
            p = stats.norm.sf(z)
        elif alternative == "less":
            cc = 0.5 if correction else 0.0
            z = (r_plus - mu + cc) / sigma
            p = stats.norm.cdf(z)
        else:
            raise ValueError("alternative must be 'two-sided', 'greater', or 'less'.")
    total = r_plus + r_minus
    rbc = (r_plus - r_minus) / total if total > 0 else 0.0
    return WilcoxonResult(float(t), float(z), float(min(p, 1.0)), float(rbc), alternative, method)


def sign_test(x: Any, y: Any = None, *, alternative: str = "two-sided") -> TestResult:
    """Sign test for paired samples (or one sample vs 0): exact binomial test on the signs of ``x - y``.

    Only the directions of the differences are used (ties dropped), so it is maximally robust but less
    powerful than the signed-rank test. ``extra`` carries ``n_positive`` and ``n`` (non-zero pairs).
    """
    _alternative(alternative)
    x = _sample("x", x)
    if y is None:
        d = x
    else:
        y_array = _sample("y", y)
        if y_array.shape != x.shape:
            raise ValueError("paired samples must have matching shapes")
        d = x - y_array
    d = d[d != 0]
    n = d.size
    n_pos = int(np.sum(d > 0))
    if n == 0:
        raise ValueError("sign test requires at least one non-zero difference")
    res = stats.binomtest(n_pos, n, 0.5, alternative=alternative)
    return TestResult(float(n_pos), float(res.pvalue), {"n_positive": n_pos, "n": n})


# --- repeated measures ------------------------------------------------------
def friedman_test(*measurements: Any) -> TestResult:
    """Friedman test for k related samples (repeated measures): the rank-based repeated-measures ANOVA.

    Pass each treatment as a separate equal-length array (one value per block). Ranks within each block,
    tie-corrects, and uses a chi-square(k-1) reference. ``extra`` carries ``df`` and Kendall's ``W``
    concordance effect size.
    """
    data = _related(measurements)
    nblocks, k = data.shape
    ranks = np.apply_along_axis(stats.rankdata, 1, data)
    rsum = ranks.sum(axis=0)
    tie = sum(_tie_term(ranks[b]) for b in range(nblocks))
    denominator = nblocks * k * (k + 1) - tie / (k - 1)
    if denominator <= 0:
        raise ValueError("Friedman test requires within-block rank variation")
    q = (12.0 * np.sum(rsum**2) - 3.0 * nblocks**2 * k * (k + 1) ** 2) / denominator
    df = k - 1
    p = float(stats.chi2.sf(q, df))
    w = q / (nblocks * (k - 1))
    return TestResult(float(q), p, {"df": df, "kendalls_w": float(w)})


# --- ordered alternatives ---------------------------------------------------
def jonckheere_terpstra(*samples: Any, alternative: str = "increasing") -> TestResult:
    """Jonckheere-Terpstra test for an ORDERED alternative across independent samples.

    More powerful than Kruskal-Wallis when the groups are expected to shift monotonically in the given
    order. ``alternative='increasing'`` / ``'decreasing'`` / ``'two-sided'``. Uses the tie-corrected
    normal approximation of the J statistic (sum of pairwise Mann-Whitney counts over ordered pairs).

    The null variance is the published Jonckheere-Terpstra one (Lehmann 1975), whose leading term is
    ``[n(n-1)(2n+5) - sum n_i(n_i-1)(2n_i+5)] / 72`` -- algebraically the same as the equivalent
    ``[n^2(2n+3) - sum n_i^2(2n_i+3)] / 72`` form. This implementation previously used
    ``n(n-1)(2n+3) - sum n_i(n_i-1)(2n_i+3)``, mixing the two (MXR-080-1599): a hybrid that matches
    neither published expression and, critically, does NOT reduce to the Mann-Whitney variance for two
    ordered groups. For two groups of five with complete separation it understated the variance as
    21.5278 against the required ``n1*n2*(n+1)/12 = 22.9167``, reporting ``z=2.69408, p=0.003529``
    where the rank-sum test gives ``z=2.61116, p=0.004512``. The tied branch inherited the same altered
    base terms, including in its ``sum t_j(t_j-1)(2t_j+5)`` tie correction.
    """
    _alternative(alternative, ("increasing", "decreasing", "two-sided"))
    groups = _groups(samples)
    k = len(groups)
    j = 0.0
    for a in range(k):
        for b in range(a + 1, k):
            j += float(np.sum(np.sign(groups[b][:, None] - groups[a][None, :]) > 0)) + 0.5 * float(
                np.sum(groups[b][:, None] == groups[a][None, :])
            )
    sizes = [g.size for g in groups]
    n = sum(sizes)
    if n < 3:
        raise ValueError("Jonckheere-Terpstra requires at least three pooled observations")
    mu = (n**2 - sum(s**2 for s in sizes)) / 4.0
    pooled = np.concatenate(groups)
    tie = _tie_term(pooled)
    # Published no-tie null variance; equals n1*n2*(n+1)/12 exactly for two ordered groups, which is
    # what pins it against the Mann-Whitney/rank-sum reference (MXR-080-1599).
    t1 = sum(s * (s - 1) * (2 * s + 5) for s in sizes)
    var = (n * (n - 1) * (2 * n + 5) - t1) / 72.0
    # tie-corrected variance (Lehmann); fall back to the no-tie form when there are no ties
    if tie > 0:
        _, tc = np.unique(pooled, return_counts=True)
        u1 = sum(c * (c - 1) * (2 * c + 5) for c in tc)
        var = (
            (n * (n - 1) * (2 * n + 5) - t1 - u1) / 72.0
            + (sum(s * (s - 1) * (s - 2) for s in sizes) * sum(c * (c - 1) * (c - 2) for c in tc))
            / (36.0 * n * (n - 1) * (n - 2))
            + (sum(s * (s - 1) for s in sizes) * sum(c * (c - 1) for c in tc)) / (8.0 * n * (n - 1))
        )
    if not np.isfinite(var) or var <= 0:
        raise ValueError("Jonckheere-Terpstra requires non-zero rank variation")
    sigma = np.sqrt(var)
    z = (j - mu) / sigma
    if alternative == "increasing":
        p = stats.norm.sf(z)
    elif alternative == "decreasing":
        p = stats.norm.cdf(z)
    elif alternative == "two-sided":
        p = 2.0 * stats.norm.sf(abs(z))
    else:
        raise ValueError("alternative must be 'increasing', 'decreasing', or 'two-sided'.")
    return TestResult(float(j), float(min(p, 1.0)), {"zscore": float(z)})


def page_trend_test(*measurements: Any, decreasing: bool = False) -> TestResult:
    """Page's trend test for an ORDERED alternative in repeated measures.

    Like Friedman but for a pre-specified ordering of the k treatments (the columns, in order). Tests
    ``L = sum_j j * R_j`` against the normal approximation. Set ``decreasing=True`` to predict the
    reverse ordering. ``extra`` carries the z-score.
    """
    data = _related(measurements)
    nblocks, k = data.shape
    ranks = np.apply_along_axis(stats.rankdata, 1, data)
    if np.all(np.ptp(ranks, axis=1) == 0.0):
        raise ValueError("Page trend test requires within-block rank variation")
    rsum = ranks.sum(axis=0)
    weights = np.arange(k, 0, -1) if decreasing else np.arange(1, k + 1)
    L = float(np.sum(weights * rsum))
    mu = nblocks * k * (k + 1) ** 2 / 4.0
    var = nblocks * k**2 * (k + 1) * (k**2 - 1) / 144.0
    z = (L - mu) / np.sqrt(var) if var > 0 else 0.0
    p = float(stats.norm.sf(z))
    return TestResult(L, float(min(p, 1.0)), {"zscore": float(z)})


# --- one-sample randomness --------------------------------------------------
def runs_test(x: Any, *, cutoff: str | float = "median") -> TestResult:
    """Wald-Wolfowitz runs test for randomness of a binary/dichotomized sequence.

    Dichotomizes ``x`` about its median (or a supplied numeric ``cutoff``) and tests whether the run
    count departs from what independence predicts (too few runs => clustering/trend; too many =>
    over-alternation). Two-sided. The null is EXACT (the closed-form run-count distribution given
    ``(n1, n2)``) when ``n1 + n2 <= 60``, where the uncorrected normal approximation is
    level-violating -- exhaustive enumeration at ``n1 = n2 = 5`` rejects 20/252 = 7.94% at nominal
    5% (STAT-RR17-16); the normal approximation applies above that. ``extra`` carries the run
    count, z-score, and ``method`` (``"exact"`` / ``"normal"``).
    """
    a = _sample("x", x, minimum=2)
    if isinstance(cutoff, str):
        if cutoff != "median":
            raise ValueError("cutoff must be 'median' or a finite number")
        c = float(np.median(a))
    else:
        c = float(cutoff)
        if not np.isfinite(c):
            raise ValueError("cutoff must be 'median' or a finite number")
    s = a[a != c] > c if cutoff == "median" else a > c
    s = np.asarray(s, dtype=bool)
    n1 = int(np.sum(s))
    n2 = int(s.size - n1)
    if n1 == 0 or n2 == 0:
        raise ValueError("runs test requires observations on both sides of the cutoff")
    runs = 1 + int(np.sum(s[1:] != s[:-1]))
    n = n1 + n2
    mu = 2.0 * n1 * n2 / n + 1.0
    var = 2.0 * n1 * n2 * (2.0 * n1 * n2 - n) / (n**2 * (n - 1))
    if var <= 0:
        raise ValueError("runs test requires enough observations for a positive reference variance")
    z = (runs - mu) / np.sqrt(var)
    if n <= 60:
        # exact conditional null given (n1, n2): the closed-form run-count pmf (STAT-RR17-16).
        # P(R = 2k) = 2 C(n1-1, k-1) C(n2-1, k-1) / C(n, n1);
        # P(R = 2k+1) = [C(n1-1, k) C(n2-1, k-1) + C(n1-1, k-1) C(n2-1, k)] / C(n, n1).
        from math import comb

        total = comb(n, n1)
        pmf = {}
        for r in range(2, n + 1):
            if r % 2 == 0:
                k = r // 2
                mass = 2 * comb(n1 - 1, k - 1) * comb(n2 - 1, k - 1)
            else:
                k = (r - 1) // 2
                mass = comb(n1 - 1, k) * comb(n2 - 1, k - 1) + comb(n1 - 1, k - 1) * comb(n2 - 1, k)
            if mass:
                pmf[r] = mass / total
        lower = sum(prob for r, prob in pmf.items() if r <= runs)
        upper = sum(prob for r, prob in pmf.items() if r >= runs)
        p = min(1.0, 2.0 * min(lower, upper))
        method = "exact"
    else:
        p = 2.0 * stats.norm.sf(abs(z))
        method = "normal"
    return TestResult(float(runs), float(min(p, 1.0)), {"runs": runs, "zscore": float(z), "method": method})


__all__ = [
    "MannWhitneyResult",
    "WilcoxonResult",
    "DunnResult",
    "TestResult",
    "mann_whitney_u",
    "cliffs_delta",
    "brunner_munzel",
    "ks_2samp",
    "ks_1samp",
    "kruskal_wallis",
    "mood_median_test",
    "dunn_test",
    "wilcoxon_signed_rank",
    "sign_test",
    "friedman_test",
    "jonckheere_terpstra",
    "page_trend_test",
    "runs_test",
]
