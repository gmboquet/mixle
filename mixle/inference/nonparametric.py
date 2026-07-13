"""Classical nonparametric (rank-based) hypothesis tests.

Distribution-free two-sample, k-sample, paired, repeated-measures, ordered-alternative, and
goodness-of-fit tests, each returning a small result object with the statistic, p-value, and -- where
standard -- an effect size. Statistics are computed here (mid-ranks for ties); tail probabilities use
the asymptotic reference distributions (normal / chi-square / Student-t / Kolmogorov) with the usual
tie and continuity corrections, matching the conventions of SciPy / R.

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


def mann_whitney_u(x: Any, y: Any, *, alternative: str = "two-sided", use_continuity: bool = True) -> MannWhitneyResult:
    """Mann-Whitney U / Wilcoxon rank-sum test for two independent samples.

    Tests whether ``x`` is stochastically greater/less than ``y``. Uses mid-ranks for ties, the
    tie-corrected normal approximation, and (default) a continuity correction. ``alternative`` is
    ``'two-sided'``, ``'greater'`` (x > y), or ``'less'``. The rank-biserial correlation
    ``2*U1/(n1 n2) - 1`` is reported as the effect size.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n1, n2 = x.size, y.size
    if n1 == 0 or n2 == 0:
        raise ValueError("both samples must be non-empty.")
    pooled = np.concatenate([x, y])
    ranks = _ranks(pooled)
    r1 = float(ranks[:n1].sum())
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    n = n1 + n2
    mu = n1 * n2 / 2.0
    sigma = np.sqrt((n1 * n2 / 12.0) * ((n + 1) - _tie_term(pooled) / (n * (n - 1))))
    if sigma == 0:
        z, p = 0.0, 1.0
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
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
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
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n1, n2 = x.size, y.size
    rank_all = _ranks(np.concatenate([x, y]))
    rx, ry = _ranks(x), _ranks(y)
    r1m, r2m = rank_all[:n1].mean(), rank_all[n1:].mean()
    s1 = np.sum((rank_all[:n1] - rx - r1m + (n1 + 1) / 2.0) ** 2) / (n1 - 1)
    s2 = np.sum((rank_all[n1:] - ry - r2m + (n2 + 1) / 2.0) ** 2) / (n2 - 1)
    denom = n1 * s1 + n2 * s2
    if denom <= 0:
        return TestResult(0.0, 1.0, {"p_hat": 0.5})
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
    """Two-sample Kolmogorov-Smirnov test: max gap between the two empirical CDFs (asymptotic p)."""
    x = np.sort(np.asarray(x, dtype=float).ravel())
    y = np.sort(np.asarray(y, dtype=float).ravel())
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
    en = n1 * n2 / (n1 + n2)
    if alternative == "two-sided":
        p = float(stats.kstwo.sf(d, int(np.round(en))))  # finite-n KS distribution (matches scipy 'asymp')
    else:
        p = float(np.exp(-2.0 * en * d * d))  # one-sided asymptotic (Smirnov)
    return TestResult(d, float(min(max(p, 0.0), 1.0)), {"n1": n1, "n2": n2})


def ks_1samp(x: Any, cdf: Callable[[np.ndarray], np.ndarray], *, alternative: str = "two-sided") -> TestResult:
    """One-sample Kolmogorov-Smirnov goodness-of-fit test against a fully-specified ``cdf`` callable."""
    x = np.sort(np.asarray(x, dtype=float).ravel())
    n = x.size
    cdfv = np.asarray(cdf(x), dtype=float)
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
    groups = [np.asarray(s, dtype=float).ravel() for s in samples]
    if len(groups) < 2:
        raise ValueError("kruskal_wallis needs at least two samples.")
    sizes = [g.size for g in groups]
    pooled = np.concatenate(groups)
    n = pooled.size
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
    groups = [np.asarray(s, dtype=float).ravel() for s in samples]
    pooled = np.concatenate(groups)
    gm = float(np.median(pooled))
    above = [int(np.sum(g > gm)) for g in groups]
    if ties == "below":
        below = [g.size - a for g, a in zip(groups, above)]
    else:  # 'above' counts ties as above
        above = [int(np.sum(g >= gm)) for g in groups]
        below = [g.size - a for g, a in zip(groups, above)]
    table = np.array([above, below], dtype=float)
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
    groups = [np.asarray(s, dtype=float).ravel() for s in samples]
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
    else:
        raise ValueError("p_adjust must be 'holm', 'bonferroni', or 'none'.")
    return DunnResult(comps, np.asarray(zs), adj, p_adjust)


# --- paired / one sample ----------------------------------------------------
@dataclass
class WilcoxonResult:
    """Result of a paired or one-sample Wilcoxon signed-rank test."""

    statistic: float  # the smaller of W+ / W- (test statistic)
    zscore: float
    pvalue: float
    rank_biserial: float
    alternative: str


def wilcoxon_signed_rank(
    x: Any, y: Any = None, *, alternative: str = "two-sided", zero_method: str = "wilcox", correction: bool = False
) -> WilcoxonResult:
    """Wilcoxon signed-rank test for paired samples (or one sample vs 0).

    Ranks ``|d|`` for ``d = x - y`` (mid-ranks for ties), splits into positive / negative rank sums, and
    uses the tie-corrected normal approximation. ``zero_method='wilcox'`` drops zero differences (and
    their ranks); ``'pratt'`` keeps them in the ranking but drops them from the sums, with the matching
    Pratt/Cureton zero corrections applied to the null mean and variance (as scipy does). The
    matched-pairs rank-biserial correlation is reported as the effect size.
    """
    x = np.asarray(x, dtype=float).ravel()
    d = x if y is None else x - np.asarray(y, dtype=float).ravel()
    if zero_method == "wilcox":
        d = d[d != 0]
    n = d.size
    if n == 0:
        return WilcoxonResult(0.0, 0.0, 1.0, 0.0, alternative)
    r = _ranks(np.abs(d))
    n_zero = 0
    if zero_method == "pratt":
        n_zero = int(np.sum(d == 0))
        keep = d != 0
        r, d = r[keep], d[keep]
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
        z, p = 0.0, 1.0
    else:
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
    return WilcoxonResult(float(t), float(z), float(min(p, 1.0)), float(rbc), alternative)


def sign_test(x: Any, y: Any = None, *, alternative: str = "two-sided") -> TestResult:
    """Sign test for paired samples (or one sample vs 0): exact binomial test on the signs of ``x - y``.

    Only the directions of the differences are used (ties dropped), so it is maximally robust but less
    powerful than the signed-rank test. ``extra`` carries ``n_positive`` and ``n`` (non-zero pairs).
    """
    x = np.asarray(x, dtype=float).ravel()
    d = x if y is None else x - np.asarray(y, dtype=float).ravel()
    d = d[d != 0]
    n = d.size
    n_pos = int(np.sum(d > 0))
    if n == 0:
        return TestResult(0.0, 1.0, {"n_positive": 0, "n": 0})
    res = stats.binomtest(n_pos, n, 0.5, alternative=alternative)
    return TestResult(float(n_pos), float(res.pvalue), {"n_positive": n_pos, "n": n})


# --- repeated measures ------------------------------------------------------
def friedman_test(*measurements: Any) -> TestResult:
    """Friedman test for k related samples (repeated measures): the rank-based repeated-measures ANOVA.

    Pass each treatment as a separate equal-length array (one value per block). Ranks within each block,
    tie-corrects, and uses a chi-square(k-1) reference. ``extra`` carries ``df`` and Kendall's ``W``
    concordance effect size.
    """
    data = np.column_stack([np.asarray(m, dtype=float).ravel() for m in measurements])
    nblocks, k = data.shape
    if k < 3:
        raise ValueError("friedman_test needs at least three related samples.")
    ranks = np.apply_along_axis(stats.rankdata, 1, data)
    rsum = ranks.sum(axis=0)
    tie = sum(_tie_term(ranks[b]) for b in range(nblocks))
    q = (12.0 * np.sum(rsum**2) - 3.0 * nblocks**2 * k * (k + 1) ** 2) / (nblocks * k * (k + 1) - tie / (k - 1))
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
    """
    groups = [np.asarray(s, dtype=float).ravel() for s in samples]
    k = len(groups)
    j = 0.0
    for a in range(k):
        for b in range(a + 1, k):
            j += float(np.sum(np.sign(groups[b][:, None] - groups[a][None, :]) > 0)) + 0.5 * float(
                np.sum(groups[b][:, None] == groups[a][None, :])
            )
    sizes = [g.size for g in groups]
    n = sum(sizes)
    mu = (n**2 - sum(s**2 for s in sizes)) / 4.0
    pooled = np.concatenate(groups)
    tie = _tie_term(pooled)
    var = (
        n * (n - 1) * (2 * n + 3)
        - sum(s * (s - 1) * (2 * s + 3) for s in sizes)
        - _tie_term(pooled) * 0  # tie adjustment folded below
    ) / 72.0
    # tie-corrected variance (Lehmann); fall back to the no-tie form when there are no ties
    if tie > 0:
        _, tc = np.unique(pooled, return_counts=True)
        t1 = sum(s * (s - 1) * (2 * s + 3) for s in sizes)
        u1 = sum(c * (c - 1) * (2 * c + 3) for c in tc)
        var = (
            (n * (n - 1) * (2 * n + 3) - t1 - u1) / 72.0
            + (sum(s * (s - 1) * (s - 2) for s in sizes) * sum(c * (c - 1) * (c - 2) for c in tc))
            / (36.0 * n * (n - 1) * (n - 2))
            + (sum(s * (s - 1) for s in sizes) * sum(c * (c - 1) for c in tc)) / (8.0 * n * (n - 1))
        )
    sigma = np.sqrt(var)
    z = (j - mu) / sigma if sigma > 0 else 0.0
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
    data = np.column_stack([np.asarray(m, dtype=float).ravel() for m in measurements])
    nblocks, k = data.shape
    ranks = np.apply_along_axis(stats.rankdata, 1, data)
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
    over-alternation). Normal approximation, two-sided. ``extra`` carries the run count and z-score.
    """
    a = np.asarray(x, dtype=float).ravel()
    c = float(np.median(a)) if cutoff == "median" else float(cutoff)
    s = a[a != c] > c if cutoff == "median" else a > c
    s = np.asarray(s, dtype=bool)
    n1 = int(np.sum(s))
    n2 = int(s.size - n1)
    if n1 == 0 or n2 == 0:
        return TestResult(1.0, 1.0, {"runs": 1, "zscore": 0.0})
    runs = 1 + int(np.sum(s[1:] != s[:-1]))
    n = n1 + n2
    mu = 2.0 * n1 * n2 / n + 1.0
    var = 2.0 * n1 * n2 * (2.0 * n1 * n2 - n) / (n**2 * (n - 1))
    z = (runs - mu) / np.sqrt(var) if var > 0 else 0.0
    p = 2.0 * stats.norm.sf(abs(z))
    return TestResult(float(runs), float(min(p, 1.0)), {"runs": runs, "zscore": float(z)})


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
