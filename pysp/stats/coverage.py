"""Sampling completeness, species richness, and diversity from frequency counts.

"How much of the population have I actually seen?" The same estimand recurs across fields: unseen
probability mass in language models, undiscovered species in ecology, unobserved rare classes in
machine learning. The answer comes from the *frequencies of frequencies* -- especially how many items
were seen exactly once (singletons) or twice (doubletons), which carry the signal about what is still
missing.

  * :func:`turing_coverage` / :func:`good_turing` -- sample coverage and the Good--Turing discounting
    that reallocates probability mass to unseen items (Good 1953; Gale & Sampson's Simple Good--Turing).
  * :func:`chao1` / :func:`chao2` -- nonparametric lower-bound richness estimators from abundance
    (Chao1) or replicated incidence (Chao2) data, with standard errors and log-normal CIs.
  * :func:`ace` / :func:`ice` -- abundance/incidence coverage-based richness estimators (rare-species
    corrected).
  * :func:`hill_numbers` -- the unified diversity profile (``q=0`` richness, ``q=1`` exp-Shannon,
    ``q=2`` inverse Simpson).
  * :func:`rarefaction_curve` -- expected richness as a function of sample size (Hurlbert
    interpolation), the basis for coverage-standardised comparison.

Counts are non-negative integer abundances per species; incidence inputs are a ``(species, sites)``
0/1 matrix.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln


def _abund(counts: np.ndarray) -> np.ndarray:
    """Coerce to a 1-D array of strictly-positive integer abundances (drop unobserved zeros)."""
    c = np.asarray(counts, dtype=float).ravel()
    if np.any(c < 0):
        raise ValueError("counts must be non-negative.")
    return c[c > 0]


def _freq_of_freq(counts: np.ndarray) -> dict[int, int]:
    """Map abundance ``r`` to the number of species observed exactly ``r`` times."""
    c = _abund(counts).astype(int)
    vals, cnts = np.unique(c, return_counts=True)
    return {int(v): int(n) for v, n in zip(vals, cnts)}


def turing_coverage(counts: np.ndarray) -> dict[str, float]:
    """Turing's sample-coverage estimate and the complementary unseen probability mass.

    ``C = 1 - f1 / n`` where ``f1`` is the number of singletons and ``n`` the total count: the
    estimated probability that the next observation is a *previously seen* species. ``1 - C = f1/n`` is
    the Good--Turing estimate of the total probability of all unseen species.

    Returns:
        ``{'coverage', 'unseen_mass', 'n', 'f1'}``.
    """
    c = _abund(counts)
    n = float(c.sum())
    f1 = float(np.sum(c == 1))
    unseen = f1 / n if n > 0 else 0.0
    return {"coverage": 1.0 - unseen, "unseen_mass": unseen, "n": n, "f1": f1}


def good_turing(counts: np.ndarray) -> dict[str, np.ndarray | float]:
    """Simple Good--Turing smoothed probabilities (Gale & Sampson 1995).

    Reallocates probability from seen to unseen items using the frequencies of frequencies. Empirical
    Turing estimates ``r* = (r+1) N_{r+1}/N_r`` are used for small ``r`` and a smoothed log-linear fit
    ``S(r)`` takes over once the two diverge (the Gale switch), giving stable discounts in the sparse
    tail.

    Args:
        counts: per-species abundances (zeros ignored).

    Returns:
        ``{'p0', 'proba', 'r_star', 'r'}`` -- ``p0`` is the total probability assigned to unseen
        species; ``proba`` are the smoothed probabilities of the *input* species (aligned to the
        positive entries of ``counts``, summing to ``1 - p0``); ``r_star`` / ``r`` are the discounted
        and raw frequencies for the distinct abundance classes.
    """
    c = _abund(counts)
    n = float(c.sum())
    fof = _freq_of_freq(c)
    r = np.array(sorted(fof), dtype=float)
    nr = np.array([fof[int(ri)] for ri in r], dtype=float)

    # Z_r: N_r divided by the half-width to the neighbouring nonzero frequencies (Gale & Sampson).
    z = np.empty_like(r)
    for i in range(len(r)):
        q = 0.0 if i == 0 else r[i - 1]
        t = 2.0 * r[i] - q if i == len(r) - 1 else r[i + 1]
        z[i] = nr[i] / (0.5 * (t - q))
    # log-linear smoothing  log Z = a + b log r
    b, a = np.polyfit(np.log(r), np.log(z), 1)
    s = lambda x: np.exp(a + b * np.log(x))  # noqa: E731

    p0 = (fof.get(1, 0) / n) if n > 0 else 0.0

    r_star = np.empty_like(r)
    use_lgt = False
    for i, ri in enumerate(r):
        lgt = (ri + 1.0) * s(ri + 1.0) / s(ri)
        next_nr = fof.get(int(ri) + 1)
        if not use_lgt and next_nr is not None:
            turing = (ri + 1.0) * next_nr / nr[i]
            se = np.sqrt((ri + 1.0) ** 2 * (next_nr / nr[i] ** 2) * (1.0 + next_nr / nr[i]))
            if abs(turing - lgt) <= 1.65 * se:
                use_lgt = True
            r_star[i] = lgt if use_lgt else turing
        else:
            use_lgt = True
            r_star[i] = lgt

    norm = float(np.sum(nr * r_star))
    rstar_of = {int(r[i]): r_star[i] for i in range(len(r))}
    proba = np.array([(1.0 - p0) * rstar_of[int(ci)] / norm for ci in c])
    return {"p0": float(p0), "proba": proba, "r_star": r_star, "r": r}


def chao1(counts: np.ndarray, *, ci_level: float = 0.95) -> dict[str, float]:
    """Chao1 nonparametric richness estimator from abundance data (bias-corrected).

    ``S_chao1 = S_obs + f1 (f1 - 1) / (2 (f2 + 1))`` (Chao 1984, bias-corrected form), a lower bound on
    total richness driven by the singleton (``f1``) and doubleton (``f2``) counts. Returns a standard
    error and a log-normal confidence interval for the number of *undetected* species (Chao 1987), so
    the interval respects ``S_chao1 >= S_obs``.

    Returns:
        ``{'estimate', 'observed', 'f1', 'f2', 'se', 'ci_low', 'ci_high'}``.
    """
    c = _abund(counts)
    s_obs = float(c.size)
    f1 = float(np.sum(c == 1))
    f2 = float(np.sum(c == 2))
    f0 = f1 * (f1 - 1.0) / (2.0 * (f2 + 1.0))
    est = s_obs + f0
    # variance of f0 (Chao 1987, bias-corrected estimator)
    t = f1 / (f2 + 1.0)
    var = 0.25 * t**2 * (2.0 * t + 1.0) ** 2 + 0.25 * t**4 - (f1**4) / (4.0 * est) if f1 > 0 else 0.0
    var = max(var, 0.0)
    se = float(np.sqrt(var))
    from scipy.stats import norm

    z = norm.ppf(0.5 + ci_level / 2.0)
    if f0 > 0 and se > 0:
        k = np.exp(z * np.sqrt(np.log(1.0 + var / f0**2)))
        ci_low = s_obs + f0 / k
        ci_high = s_obs + f0 * k
    else:
        ci_low = ci_high = est
    return {
        "estimate": est,
        "observed": s_obs,
        "f1": f1,
        "f2": f2,
        "se": se,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def chao2(incidence: np.ndarray, *, ci_level: float = 0.95) -> dict[str, float]:
    """Chao2 richness estimator from replicated incidence (presence/absence) data.

    Args:
        incidence: ``(n_species, n_sites)`` 0/1 matrix (or per-species counts of sites occupied,
            passed as a 1-D array together with ``... `` -- a 2-D matrix is expected here).
        ci_level: confidence level for the log-normal interval.

    Returns:
        ``{'estimate', 'observed', 'q1', 'q2', 'se', 'ci_low', 'ci_high', 'sites'}`` where ``q1``/``q2``
        are the numbers of species found in exactly one / two sites.
    """
    inc = np.atleast_2d(np.asarray(incidence, dtype=float))
    inc = (inc > 0).astype(int)
    site_counts = inc.sum(axis=1)
    site_counts = site_counts[site_counts > 0]
    m = float(inc.shape[1])
    s_obs = float(site_counts.size)
    q1 = float(np.sum(site_counts == 1))
    q2 = float(np.sum(site_counts == 2))
    corr = (m - 1.0) / m
    q0 = corr * q1 * (q1 - 1.0) / (2.0 * (q2 + 1.0))
    est = s_obs + q0
    t = q1 / (q2 + 1.0)
    var = max(0.25 * corr * t**2 * (2.0 * t + 1.0) ** 2 + 0.25 * corr**2 * t**4, 0.0) if q1 > 0 else 0.0
    se = float(np.sqrt(var))
    from scipy.stats import norm

    z = norm.ppf(0.5 + ci_level / 2.0)
    if q0 > 0 and se > 0:
        k = np.exp(z * np.sqrt(np.log(1.0 + var / q0**2)))
        ci_low, ci_high = s_obs + q0 / k, s_obs + q0 * k
    else:
        ci_low = ci_high = est
    return {
        "estimate": est,
        "observed": s_obs,
        "q1": q1,
        "q2": q2,
        "se": se,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "sites": m,
    }


def ace(counts: np.ndarray, *, rare_threshold: int = 10) -> dict[str, float]:
    """ACE: Abundance-based Coverage Estimator of richness (Chao & Lee 1992).

    Splits species into abundant (``> rare_threshold``) and rare, estimates sample coverage from the
    rare group's singletons, and corrects for the coefficient of variation of the rare abundances.

    Returns:
        ``{'estimate', 'observed', 's_rare', 's_abund', 'c_ace'}``.
    """
    c = _abund(counts)
    s_obs = float(c.size)
    rare = c[c <= rare_threshold]
    abund = c[c > rare_threshold]
    s_rare = float(rare.size)
    s_abund = float(abund.size)
    n_rare = float(rare.sum())
    f1 = float(np.sum(c == 1))
    c_ace = 1.0 - f1 / n_rare if n_rare > 0 else 1.0
    if c_ace <= 0 or s_rare == 0:
        return {"estimate": s_obs, "observed": s_obs, "s_rare": s_rare, "s_abund": s_abund, "c_ace": c_ace}
    sum_ii = float(np.sum(np.array([i * (i - 1) for i in rare])))
    gamma2 = max((s_rare / c_ace) * sum_ii / (n_rare * (n_rare - 1.0)) - 1.0, 0.0)
    est = s_abund + s_rare / c_ace + (f1 / c_ace) * gamma2
    return {"estimate": float(est), "observed": s_obs, "s_rare": s_rare, "s_abund": s_abund, "c_ace": float(c_ace)}


def ice(incidence: np.ndarray, *, rare_threshold: int = 10) -> dict[str, float]:
    """ICE: Incidence-based Coverage Estimator of richness (the Chao--Lee estimator for incidence data).

    Args:
        incidence: ``(n_species, n_sites)`` 0/1 matrix.
        rare_threshold: species found in ``<= rare_threshold`` sites are treated as infrequent.

    Returns:
        ``{'estimate', 'observed', 's_infreq', 's_freq', 'c_ice'}``.
    """
    inc = (np.atleast_2d(np.asarray(incidence, dtype=float)) > 0).astype(int)
    site_counts = inc.sum(axis=1)
    site_counts = site_counts[site_counts > 0]
    s_obs = float(site_counts.size)
    infreq = site_counts[site_counts <= rare_threshold]
    freq = site_counts[site_counts > rare_threshold]
    s_infreq = float(infreq.size)
    s_freq = float(freq.size)
    n_infreq = float(infreq.sum())
    q1 = float(np.sum(site_counts == 1))
    n_sites = float(inc.shape[1])
    c_ice = 1.0 - q1 / n_infreq if n_infreq > 0 else 1.0
    if c_ice <= 0 or s_infreq == 0:
        return {"estimate": s_obs, "observed": s_obs, "s_infreq": s_infreq, "s_freq": s_freq, "c_ice": c_ice}
    sum_jj = float(np.sum(np.array([j * (j - 1) for j in infreq])))
    factor = n_sites / (n_sites - 1.0) if n_sites > 1 else 1.0
    gamma2 = max((s_infreq / c_ice) * factor * sum_jj / (n_infreq * (n_infreq - 1.0)) - 1.0, 0.0)
    est = s_freq + s_infreq / c_ice + (q1 / c_ice) * gamma2
    return {"estimate": float(est), "observed": s_obs, "s_infreq": s_infreq, "s_freq": s_freq, "c_ice": float(c_ice)}


def hill_numbers(counts: np.ndarray, q: float | np.ndarray = (0.0, 1.0, 2.0)) -> np.ndarray:
    """Hill numbers (effective number of species) of order ``q``.

    The unified diversity profile: ``q=0`` is observed richness, ``q=1`` is the exponential of Shannon
    entropy, ``q=2`` is the inverse Simpson concentration. Larger ``q`` weights common species more, so
    the profile ``D(q)`` summarises evenness as well as richness.

    Args:
        counts: per-species abundances.
        q: a scalar order or an array of orders.

    Returns:
        Array of Hill numbers, one per requested order (scalar input still returns a length-1 array).
    """
    c = _abund(counts)
    p = c / c.sum()
    qs = np.atleast_1d(np.asarray(q, dtype=float))
    out = np.empty(qs.shape[0])
    for i, qi in enumerate(qs):
        if np.isclose(qi, 1.0):
            out[i] = float(np.exp(-np.sum(p * np.log(p))))
        else:
            out[i] = float(np.sum(p**qi) ** (1.0 / (1.0 - qi)))
    return out


def rarefaction_curve(counts: np.ndarray, sizes: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Individual-based rarefaction: expected richness when subsampling ``m`` individuals (Hurlbert).

    ``E[S(m)] = sum_i (1 - C(n - x_i, m) / C(n, m))`` -- the expected number of species seen in a random
    subsample of ``m`` of the ``n`` individuals. Used to compare richness between samples at a common
    sample size (or coverage).

    Args:
        counts: per-species abundances.
        sizes: subsample sizes ``m`` to evaluate; defaults to ``1 .. n``.

    Returns:
        ``{'sizes', 'expected_richness'}``.
    """
    c = _abund(counts).astype(int)
    n = int(c.sum())
    if sizes is None:
        sizes = np.arange(1, n + 1)
    sizes = np.asarray(sizes, dtype=int)
    ln_choose_n = gammaln(n + 1) - gammaln(np.arange(n + 1) + 1) - gammaln(n - np.arange(n + 1) + 1)

    def log_choose(a: int, m: int) -> float:
        if m < 0 or m > a:
            return -np.inf
        return gammaln(a + 1) - gammaln(m + 1) - gammaln(a - m + 1)

    exp_rich = np.empty(sizes.shape[0], dtype=float)
    for j, m in enumerate(sizes):
        denom = ln_choose_n[m]
        miss = 0.0
        for xi in c:
            miss += np.exp(log_choose(n - xi, m) - denom)
        exp_rich[j] = c.size - miss
    return {"sizes": sizes, "expected_richness": exp_rich}


__all__ = [
    "turing_coverage",
    "good_turing",
    "chao1",
    "chao2",
    "ace",
    "ice",
    "hill_numbers",
    "rarefaction_curve",
]
