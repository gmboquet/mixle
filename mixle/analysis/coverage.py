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

Counts must be a one-dimensional vector of finite non-negative integer abundances per species
(validated by :func:`_abund`, which raises rather than silently coercing values or flattening axes);
incidence inputs must be a finite binary ``(species, sites)`` 0/1 matrix with at least one site
(validated by :func:`_incidence`). Estimators raise :class:`CoverageInsufficientDataError` rather than
issuing contradictory numerical answers when no observations are present.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np
from scipy.special import gammaln


class CoverageInsufficientDataError(ValueError):
    """The requested coverage/diversity estimand is not identified by an empty sample."""


def _abund(counts: np.ndarray) -> np.ndarray:
    """Validate and coerce to a 1-D array of non-negative integer abundances (drop unobserved zeros).

    Every abundance-based estimator in this module routes its counts through here, so "abundance" has
    exactly one meaning throughout: a finite, non-negative, exact integer per species. Fractional,
    negative, and non-finite counts are all rejected with a clear error instead of being silently
    reinterpreted -- fractional counts used to be summed as continuous totals by some estimators
    (``turing_coverage``) and truncated to integer abundance *classes* by others (``good_turing`` via
    ``_freq_of_freq``), two incompatible readings of the same input such as ``[1.5, 2.5]``. NaN used to
    be dropped indirectly by comparisons that NaN always fails (``NaN > 0`` is ``False``), silently
    shrinking the effective sample with no signal (MXR-080-0077).
    """
    raw = np.asarray(counts)
    if raw.ndim != 1:
        raise ValueError(f"counts must be a one-dimensional species-abundance vector, got shape {raw.shape}.")
    maximum = np.iinfo(np.int64).max
    validated: list[int] = []
    for value in raw.tolist():
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("counts must be integer abundances, not Boolean values.")
        if isinstance(value, Integral):
            count = int(value)
        elif isinstance(value, Real):
            numeric = float(value)
            if not np.isfinite(numeric):
                raise ValueError("counts must be finite (no NaN or Inf).")
            if not numeric.is_integer():
                raise ValueError("counts must be exact integers (fractional abundances are not supported).")
            count = int(numeric)
        else:
            raise TypeError(f"counts must contain numeric integer abundances, got {value!r}.")
        if count < 0:
            raise ValueError("counts must be non-negative.")
        if count > maximum:
            raise OverflowError(f"an abundance exceeds the supported signed 64-bit per-species limit ({maximum}).")
        if count:
            validated.append(count)
    return np.asarray(validated, dtype=np.int64)


def _abundance_total(counts: np.ndarray) -> int:
    """Return the exact Python-integer total without NumPy signed-integer overflow."""
    return sum(int(count) for count in counts)


def _require_observed(counts: np.ndarray, estimator: str) -> None:
    if counts.size == 0:
        raise CoverageInsufficientDataError(
            f"{estimator} requires at least one observed individual or incidence; an empty sample "
            "does not identify coverage, unseen mass, richness, or diversity."
        )


def _ci_level(value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("ci_level must be a scalar probability, not a Boolean or array.")
    level = float(value)
    if not np.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("ci_level must be finite and strictly between 0 and 1.")
    return level


def _rare_threshold(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("rare_threshold must be an exact non-Boolean integer.")
    threshold = int(value)
    if threshold < 1:
        raise ValueError("rare_threshold must be at least 1.")
    return threshold


def _freq_of_freq(counts: np.ndarray) -> dict[int, int]:
    """Map abundance ``r`` to the number of species observed exactly ``r`` times."""
    c = _abund(counts)
    vals, cnts = np.unique(c, return_counts=True)
    return {int(v): int(n) for v, n in zip(vals, cnts)}


def _incidence(matrix: np.ndarray) -> np.ndarray:
    """Validate a finite binary presence/absence matrix and coerce it to ``(n_species, n_sites)`` int.

    Every entry must be exactly ``0`` or ``1``. Fractional (``0.2``), out-of-range (``3``), negative
    (``-1``), and non-finite (``NaN``) entries are all rejected rather than silently thresholded to
    presence/absence -- any positive value used to become ``1`` and any negative/NaN value used to
    become ``0``, so a matrix of continuous or malformed measurements was silently accepted as a valid
    incidence design (MXR-080-0079). At least one site (column) is required.
    """
    inc = np.asarray(matrix)
    if inc.ndim != 2:
        raise ValueError(f"incidence must be a two-dimensional (species, sites) matrix, got shape {inc.shape}.")
    if inc.shape[1] == 0:
        raise ValueError("incidence matrix must have at least one site.")
    try:
        finite = np.isfinite(inc)
    except TypeError as exc:
        raise TypeError("incidence matrix entries must be numeric Boolean/binary values.") from exc
    if not np.all(finite):
        raise ValueError("incidence matrix must be finite (no NaN or Inf).")
    if not np.all((inc == 0) | (inc == 1)):
        raise ValueError("incidence matrix must be binary (every entry must be exactly 0 or 1).")
    return inc.astype(np.int64)


def turing_coverage(counts: np.ndarray) -> dict[str, float | int]:
    """Turing's sample-coverage estimate and the complementary unseen probability mass.

    ``C = 1 - f1 / n`` where ``f1`` is the number of singletons and ``n`` the total count: the
    estimated probability that the next observation is a *previously seen* species. ``1 - C = f1/n`` is
    the Good--Turing estimate of the total probability of all unseen species.

    Returns:
        ``{'coverage', 'unseen_mass', 'n', 'f1'}``.
    """
    c = _abund(counts)
    _require_observed(c, "turing_coverage")
    n = _abundance_total(c)
    f1 = int(np.sum(c == 1))
    unseen = f1 / n
    return {"coverage": 1.0 - unseen, "unseen_mass": unseen, "n": n, "f1": f1}


def good_turing(counts: np.ndarray) -> dict[str, np.ndarray | float | bool | str]:
    """Simple Good--Turing smoothed probabilities (Gale & Sampson 1995).

    Reallocates probability from seen to unseen items using the frequencies of frequencies. Empirical
    Turing estimates ``r* = (r+1) N_{r+1}/N_r`` are used for small ``r`` and a smoothed log-linear fit
    ``S(r)`` takes over once the two diverge (the Gale switch), giving stable discounts in the sparse
    tail.

    The log-linear fit needs at least two distinct abundance classes to be identifiable. With exactly
    one -- e.g. an all-singleton sample like ``[1, 1, 1]``, a canonical Good--Turing input, not a
    malformed one -- there is nothing to fit a slope against, so smoothing is skipped and the seen mass
    is reallocated by raw frequency instead (the Simple Good--Turing small-support fallback; previously
    this raised a ``LinAlgError`` out of ``np.polyfit``). With no observed counts at all, coverage is
    not identifiable and the common :class:`CoverageInsufficientDataError` is raised.

    Args:
        counts: per-species abundances (zeros ignored).

    Returns:
        ``{'p0', 'proba', 'r_star', 'r', 'insufficient_evidence', 'reason'}`` -- ``p0`` is the total
        probability assigned to unseen species; ``proba`` are the smoothed probabilities of the
        *input* species (aligned to the positive entries of ``counts``, summing to ``1 - p0``);
        ``r_star`` / ``r`` are the discounted and raw frequencies for the distinct abundance classes.
        ``insufficient_evidence`` is always ``False`` for a returned estimate; empty samples raise
        :class:`CoverageInsufficientDataError`.
    """
    c = _abund(counts)
    _require_observed(c, "good_turing")
    n = _abundance_total(c)
    fof = _freq_of_freq(c)
    r = np.array(sorted(fof), dtype=float)
    nr = np.array([fof[int(ri)] for ri in r], dtype=float)
    p0 = fof.get(1, 0) / n

    if len(r) < 2:
        # Simple Good-Turing small-support fallback (Gale & Sampson 1995): the log-linear regression
        # log Z = a + b log r needs >= 2 distinct (r, Z_r) points to determine a slope -- a single
        # point (e.g. every observed species a singleton) is a well-posed ecological sample but an
        # underdetermined regression. With nothing to smooth against, skip smoothing and reallocate
        # the seen mass by raw frequency (r* = r), equivalent to the ordinary renormalized MLE over
        # the seen species.
        r_star = r.copy()
    else:
        # Z_r: N_r divided by the half-width to the neighbouring nonzero frequencies (Gale & Sampson).
        z = np.empty_like(r)
        for i in range(len(r)):
            q = 0.0 if i == 0 else r[i - 1]
            t = 2.0 * r[i] - q if i == len(r) - 1 else r[i + 1]
            z[i] = nr[i] / (0.5 * (t - q))
        # log-linear smoothing  log Z = a + b log r
        b, a = np.polyfit(np.log(r), np.log(z), 1)
        s = lambda x: np.exp(a + b * np.log(x))  # noqa: E731

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
    return {
        "p0": float(p0),
        "proba": proba,
        "r_star": r_star,
        "r": r,
        "insufficient_evidence": False,
        "reason": "",
    }


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
    _require_observed(c, "chao1")
    ci_level = _ci_level(ci_level)
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
        incidence: finite binary ``(n_species, n_sites)`` 0/1 matrix (validated by :func:`_incidence`;
            every entry must be exactly 0 or 1, and at least one site is required).
        ci_level: confidence level for the log-normal interval.

    Returns:
        ``{'estimate', 'observed', 'q1', 'q2', 'se', 'ci_low', 'ci_high', 'sites'}`` where ``q1``/``q2``
        are the numbers of species found in exactly one / two sites.
    """
    inc = _incidence(incidence)
    ci_level = _ci_level(ci_level)
    site_counts = inc.sum(axis=1)
    site_counts = site_counts[site_counts > 0]
    _require_observed(site_counts, "chao2")
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
    _require_observed(c, "ace")
    rare_threshold = _rare_threshold(rare_threshold)
    s_obs = float(c.size)
    rare = c[c <= rare_threshold]
    abund = c[c > rare_threshold]
    s_rare = float(rare.size)
    s_abund = float(abund.size)
    n_rare = float(_abundance_total(rare))
    f1 = float(np.sum(c == 1))
    c_ace = 1.0 - f1 / n_rare if n_rare > 0 else 1.0
    if c_ace <= 0 or s_rare == 0:
        return {"estimate": s_obs, "observed": s_obs, "s_rare": s_rare, "s_abund": s_abund, "c_ace": c_ace}
    sum_ii = math.fsum(float(i) * (float(i) - 1.0) for i in rare)
    gamma2 = max((s_rare / c_ace) * sum_ii / (n_rare * (n_rare - 1.0)) - 1.0, 0.0)
    est = s_abund + s_rare / c_ace + (f1 / c_ace) * gamma2
    return {"estimate": float(est), "observed": s_obs, "s_rare": s_rare, "s_abund": s_abund, "c_ace": float(c_ace)}


def ice(incidence: np.ndarray, *, rare_threshold: int = 10) -> dict[str, float]:
    """ICE: Incidence-based Coverage Estimator of richness (the Chao--Lee estimator for incidence data).

    Args:
        incidence: finite binary ``(n_species, n_sites)`` 0/1 matrix (validated by :func:`_incidence`;
            every entry must be exactly 0 or 1, and at least one site is required).
        rare_threshold: species found in ``<= rare_threshold`` sites are treated as infrequent.

    Returns:
        ``{'estimate', 'observed', 's_infreq', 's_freq', 'c_ice'}``.
    """
    inc = _incidence(incidence)
    rare_threshold = _rare_threshold(rare_threshold)
    site_counts = inc.sum(axis=1)
    site_counts = site_counts[site_counts > 0]
    _require_observed(site_counts, "ice")
    s_obs = float(site_counts.size)
    infreq = site_counts[site_counts <= rare_threshold]
    freq = site_counts[site_counts > rare_threshold]
    s_infreq = float(infreq.size)
    s_freq = float(freq.size)
    n_infreq = float(_abundance_total(infreq))
    q1 = float(np.sum(site_counts == 1))
    n_sites = float(inc.shape[1])
    c_ice = 1.0 - q1 / n_infreq if n_infreq > 0 else 1.0
    if c_ice <= 0 or s_infreq == 0:
        return {"estimate": s_obs, "observed": s_obs, "s_infreq": s_infreq, "s_freq": s_freq, "c_ice": c_ice}
    sum_jj = math.fsum(float(j) * (float(j) - 1.0) for j in infreq)
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
    _require_observed(c, "hill_numbers")
    raw_q = np.asarray(q)
    if raw_q.ndim > 1:
        raise ValueError(f"q must be a scalar or one-dimensional order vector, got shape {raw_q.shape}.")
    if raw_q.dtype.kind == "b":
        raise TypeError("q must contain numeric diversity orders, not Boolean values.")
    if raw_q.dtype.kind not in "iuf":
        raise TypeError("q must contain real numeric diversity orders.")
    try:
        qs = np.atleast_1d(raw_q.astype(float))
    except (TypeError, ValueError) as exc:
        raise TypeError("q must contain numeric diversity orders.") from exc
    if qs.size == 0 or not np.all(np.isfinite(qs)) or np.any(qs < 0.0):
        raise ValueError("q must contain at least one finite non-negative diversity order.")
    p = c.astype(np.float64) / float(_abundance_total(c))
    out = np.empty(qs.shape[0])
    for i, qi in enumerate(qs):
        if np.isclose(qi, 1.0):
            out[i] = float(np.exp(-np.sum(p * np.log(p))))
        else:
            out[i] = float(np.sum(p**qi) ** (1.0 / (1.0 - qi)))
    return out


def _rarefaction_sizes(sizes: np.ndarray, n: int) -> np.ndarray:
    """Validate rarefaction subsample sizes as exact integers in ``[0, n]``, preserving caller order.

    Sizes used to be cast with a bare ``dtype=int``: a fractional size (``1.9``) silently truncated
    instead of raising, a negative size (``-1``) silently returned a "plausible" richness via negative
    array indexing instead of raising, and a size above the sample count ``n`` raised an internal
    ``IndexError`` deep inside the routine instead of a clear domain error naming the valid range
    (MXR-080-0080). No sorting or deduplication happens here -- the returned array has the same length
    and order as ``sizes``.
    """
    raw = np.asarray(sizes)
    if raw.ndim != 1:
        raise ValueError(f"rarefaction sizes must be a one-dimensional vector, got shape {raw.shape}.")
    validated: list[int] = []
    for value in raw.tolist():
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("rarefaction sizes must be exact non-Boolean integers.")
        if isinstance(value, Integral):
            size = int(value)
        elif isinstance(value, Real):
            numeric = float(value)
            if not np.isfinite(numeric):
                raise ValueError("rarefaction sizes must be finite (no NaN or Inf).")
            if not numeric.is_integer():
                raise ValueError("rarefaction sizes must be exact integers (fractional sizes are not supported).")
            size = int(numeric)
        else:
            raise TypeError(f"rarefaction sizes must be numeric integers, got {value!r}.")
        if not 0 <= size <= n:
            raise ValueError(f"rarefaction sizes must be within [0, {n}], got {size}.")
        validated.append(size)
    dtype = object if any(size > np.iinfo(np.int64).max for size in validated) else np.int64
    return np.asarray(validated, dtype=dtype)


def rarefaction_curve(counts: np.ndarray, sizes: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Individual-based rarefaction: expected richness when subsampling ``m`` individuals (Hurlbert).

    ``E[S(m)] = sum_i (1 - C(n - x_i, m) / C(n, m))`` -- the expected number of species seen in a random
    subsample of ``m`` of the ``n`` individuals. Used to compare richness between samples at a common
    sample size (or coverage).

    Args:
        counts: per-species abundances.
        sizes: subsample sizes ``m`` to evaluate; defaults to ``1 .. n``. Must be exact integers in
            ``[0, n]`` (validated by :func:`_rarefaction_sizes`); the result is returned in the same
            order as ``sizes`` (not sorted or deduplicated).

    Returns:
        ``{'sizes', 'expected_richness'}``.
    """
    c = _abund(counts)
    _require_observed(c, "rarefaction_curve")
    n = _abundance_total(c)
    if sizes is None:
        if n > 1_000_000:
            raise ValueError(
                "the default rarefaction grid would exceed 1,000,000 points; provide explicit sizes."
            )
        sizes = np.arange(1, n + 1)
    else:
        sizes = _rarefaction_sizes(sizes, n)

    def log_choose(a: int, m: int) -> float:
        if m < 0 or m > a:
            return -np.inf
        k = min(m, a - m)
        if k <= 10_000:
            return math.fsum(math.log(a - k + i) - math.log(i) for i in range(1, k + 1))
        return gammaln(a + 1) - gammaln(m + 1) - gammaln(a - m + 1)

    exp_rich = np.empty(sizes.shape[0], dtype=float)
    for j, m in enumerate(sizes):
        m = int(m)
        denom = log_choose(n, m)
        miss = 0.0
        for xi in c:
            miss += np.exp(log_choose(n - int(xi), m) - denom)
        exp_rich[j] = c.size - miss
    return {"sizes": sizes, "expected_richness": exp_rich}


__all__ = [
    "CoverageInsufficientDataError",
    "turing_coverage",
    "good_turing",
    "chao1",
    "chao2",
    "ace",
    "ice",
    "hill_numbers",
    "rarefaction_curve",
]
