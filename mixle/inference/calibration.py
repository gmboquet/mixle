"""Calibration diagnostics: "is my probability / interval actually calibrated?"

A forecast is *calibrated* when its stated probabilities match observed frequencies: events given
70% should happen ~70% of the time, and 90% intervals should contain the truth ~90% of the time. This
is distinct from *sharpness* (how concentrated the forecast is) and from *accuracy* -- a forecaster
can be perfectly calibrated while carrying little decision value (for example, always predicting the
base rate), so calibration is a necessary, not sufficient, condition that you check separately. These
diagnostics are model-free: they look only at predicted probabilities/intervals and what happened.

Three families, by forecast type:

  * **Probability classifiers** -- :func:`reliability_curve` (the reliability diagram), and the
    :func:`expected_calibration_error` / :func:`maximum_calibration_error` summaries. For multiclass
    use :func:`top_label_confidence` to reduce to the (confidence, correct) calibration problem.
  * **Full predictive distributions** -- the Probability Integral Transform: :func:`pit_values` /
    :func:`pit_ensemble`, the :func:`pit_histogram`, and :func:`pit_calibration_error`. Under a
    calibrated forecast the PIT values are Uniform(0, 1); a U-shaped histogram means under-dispersion,
    a hump means over-dispersion, a slope means bias.
  * **Intervals / quantiles** -- :func:`interval_coverage` (coverage and mean width at one level) and
    :func:`coverage_curve` (empirical-vs-nominal coverage across a grid of levels).

Several functions take ``ci=True`` to attach nonparametric bootstrap intervals. These are
PER-BIN POINTWISE intervals, not a simultaneous band: each bin's interval covers that bin's own
frequency at the stated level, but the chance that at least one of ``bins`` intervals excludes its
truth is far higher (measured 30-45% familywise exclusion at a nominal 5% per bin on reliability
diagrams) -- so one bin poking outside its interval is expected under perfect calibration, not
evidence against it. The binned summaries also have a NULL FLOOR: with finitely many outcomes per
bin, ECE and MCE are strictly positive even for a perfectly calibrated forecaster; compare them to
:func:`calibration_null_expectation`, not to zero.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.random import RandomState


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _vector(name: str, values: Any) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-empty finite numeric vector") from exc
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _probability_outcomes(prob: Any, outcome: Any) -> tuple[np.ndarray, np.ndarray]:
    p = _vector("prob", prob)
    y = _vector("outcome", outcome)
    if p.shape != y.shape:
        raise ValueError("prob and outcome must have matching shapes")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("prob must contain probabilities in [0, 1]")
    if np.any((y != 0.0) & (y != 1.0)):
        raise ValueError("outcome must contain binary 0/1 values")
    return p, y


def _pit_vector(values: Any) -> np.ndarray:
    pit = _vector("pit", values)
    if np.any((pit < 0.0) | (pit > 1.0)):
        raise ValueError("PIT/CDF values must lie in [0, 1]")
    return pit


def _bootstrap_controls(n_boot: int, ci_level: float) -> tuple[int, float]:
    n_boot = _positive_int("n_boot", n_boot)
    if (
        isinstance(ci_level, (bool, np.bool_))
        or not isinstance(ci_level, (int, float, np.integer, np.floating))
        or not np.isfinite(ci_level)
        or not 0.0 < float(ci_level) < 1.0
    ):
        raise ValueError("ci_level must be a finite number strictly between 0 and 1")
    return n_boot, float(ci_level)


def _as_rng(seed: int | RandomState | None) -> RandomState:
    """Return a ``RandomState`` from an int seed, an existing ``RandomState``, or ``None``."""
    if isinstance(seed, RandomState):
        return seed
    return RandomState(seed)


def _bin_edges(prob: np.ndarray, bins: int, strategy: str) -> np.ndarray:
    """Equal-width (``"uniform"``) or equal-count (``"quantile"``) bin edges on ``[0, 1]``."""
    bins = _positive_int("bins", bins)
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, bins + 1)
    if strategy == "quantile":
        edges = np.quantile(prob, np.linspace(0.0, 1.0, bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        return np.unique(edges)
    raise ValueError("strategy must be 'uniform' or 'quantile'.")


def reliability_curve(
    prob: np.ndarray,
    outcome: np.ndarray,
    *,
    bins: int = 10,
    strategy: str = "uniform",
    ci: bool = False,
    n_boot: int = 1000,
    ci_level: float = 0.95,
    seed: int | RandomState | None = 0,
) -> dict[str, np.ndarray]:
    """Reliability diagram: observed frequency vs mean forecast probability, per bin.

    Bins the forecasts, then within each bin compares the mean predicted probability to the observed
    event frequency. A perfectly calibrated forecaster lies on the diagonal ``observed == predicted``.

    Args:
        prob: ``(n,)`` predicted probabilities of the positive class (or top-label confidences).
        outcome: ``(n,)`` 0/1 outcomes (or correctness indicators).
        bins: number of bins.
        strategy: ``"uniform"`` (equal-width) or ``"quantile"`` (equal-count) bins.
        ci: if True attach a percentile bootstrap interval on the observed frequency in each bin.
        n_boot: bootstrap resamples when ``ci`` is True.
        ci_level: central probability of each PER-BIN interval (e.g. 0.95).
        seed: RNG seed for the bootstrap.

    Returns:
        ``{'mean_pred', 'obs_freq', 'count', 'bin_edges'}`` (one entry per non-empty bin), plus
        ``'obs_lo'`` / ``'obs_hi'`` / ``'obs_ci_effective_boot'`` when ``ci`` is True.

    The intervals are POINTWISE, one bin at a time -- with 10 bins at 95% each, at least one bin
    escaping its interval under perfect calibration is a ~30-45% event, so judge single-bin
    excursions accordingly. A resample in which a bin comes up EMPTY contributes nothing to that
    bin's interval; the quantiles are taken over the non-empty resamples only, which conditions the
    interval on the bin being observed (a real bias for sparse bins). ``obs_ci_effective_boot``
    reports, per bin, how many of the ``n_boot`` resamples actually informed the interval -- treat
    an interval resting on far fewer than ``n_boot`` replicates as unstable rather than narrow.
    """
    p, y = _probability_outcomes(prob, outcome)
    if ci:
        n_boot, ci_level = _bootstrap_controls(n_boot, ci_level)
    edges = _bin_edges(p, bins, strategy)
    nb = len(edges) - 1
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, nb - 1)

    mean_pred, obs_freq, count, used = [], [], [], []
    for b in range(nb):
        mask = idx == b
        c = int(mask.sum())
        if c == 0:
            continue
        mean_pred.append(float(p[mask].mean()))
        obs_freq.append(float(y[mask].mean()))
        count.append(c)
        used.append(b)
    out = {
        "mean_pred": np.asarray(mean_pred),
        "obs_freq": np.asarray(obs_freq),
        "count": np.asarray(count, dtype=int),
        "bin_edges": edges,
    }
    if not ci:
        return out

    rng = _as_rng(seed)
    n = p.shape[0]
    boot = np.full((n_boot, len(used)), np.nan)
    for r in range(n_boot):
        sel = rng.randint(0, n, size=n)
        ps, ys = p[sel], y[sel]
        bidx = np.clip(np.digitize(ps, edges[1:-1], right=False), 0, nb - 1)
        for j, b in enumerate(used):
            m = bidx == b
            if m.any():
                boot[r, j] = ys[m].mean()
    lo_q = (1.0 - ci_level) / 2.0
    out["obs_lo"] = np.nanquantile(boot, lo_q, axis=0)
    out["obs_hi"] = np.nanquantile(boot, 1.0 - lo_q, axis=0)
    # NaN rows are resamples where the bin was empty: nanquantile drops them, so the interval is
    # conditioned on the bin existing. Surface how many replicates each interval actually rests on.
    out["obs_ci_effective_boot"] = np.sum(~np.isnan(boot), axis=0).astype(int)
    return out


def expected_calibration_error(
    prob: np.ndarray,
    outcome: np.ndarray,
    *,
    bins: int = 10,
    strategy: str = "uniform",
    norm: str = "l1",
    ci: bool = False,
    n_boot: int = 1000,
    ci_level: float = 0.95,
    seed: int | RandomState | None = 0,
) -> float | tuple[float, float, float]:
    """Expected Calibration Error: count-weighted average gap between confidence and accuracy.

    ``ECE = sum_b (n_b / n) |obs_b - pred_b|`` over bins (``norm='l2'`` uses the squared gap, square-
    rooted). For multiclass classifiers reduce with :func:`top_label_confidence` first.

    Zero is NOT the null value. The plug-in ECE takes an absolute value of binomial noise in every
    bin, so a PERFECTLY calibrated forecaster still scores roughly ``sqrt(bins / n)`` in expectation
    -- and a percentile bootstrap interval around the plug-in estimate inherits that upward shift,
    so under perfect calibration it excludes the true value 0 essentially always. Judge the point
    estimate against :func:`calibration_null_expectation` (the measured perfect-calibration
    distribution of this same statistic at your ``prob`` profile and binning), and read the
    bootstrap interval as sampling uncertainty around the BIASED plug-in functional, not as a test
    of calibration.

    Args:
        prob: ``(n,)`` predicted probabilities / confidences.
        outcome: ``(n,)`` 0/1 outcomes / correctness indicators.
        bins: number of bins.
        strategy: ``"uniform"`` or ``"quantile"`` binning.
        norm: ``"l1"`` (mean absolute gap) or ``"l2"`` (root mean squared gap).
        ci: if True also return a percentile bootstrap interval.
        n_boot, ci_level, seed: bootstrap controls.

    Returns:
        The ECE (float), or ``(ece, lo, hi)`` when ``ci`` is True.
    """

    def _ece(p: np.ndarray, y: np.ndarray) -> float:
        edges = _bin_edges(p, bins, strategy)
        nb = len(edges) - 1
        idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, nb - 1)
        n = p.shape[0]
        total = 0.0
        for b in range(nb):
            mask = idx == b
            c = int(mask.sum())
            if c == 0:
                continue
            gap = abs(float(y[mask].mean()) - float(p[mask].mean()))
            total += (c / n) * (gap if norm == "l1" else gap * gap)
        if norm == "l2":
            return float(np.sqrt(total))
        if norm != "l1":
            raise ValueError("norm must be 'l1' or 'l2'.")
        return float(total)

    p, y = _probability_outcomes(prob, outcome)
    if norm not in ("l1", "l2"):
        raise ValueError("norm must be 'l1' or 'l2'.")
    if ci:
        n_boot, ci_level = _bootstrap_controls(n_boot, ci_level)
    point = _ece(p, y)
    if not ci:
        return point
    rng = _as_rng(seed)
    n = p.shape[0]
    boot = np.empty(n_boot)
    for r in range(n_boot):
        sel = rng.randint(0, n, size=n)
        boot[r] = _ece(p[sel], y[sel])
    lo_q = (1.0 - ci_level) / 2.0
    return point, float(np.quantile(boot, lo_q)), float(np.quantile(boot, 1.0 - lo_q))


def maximum_calibration_error(
    prob: np.ndarray, outcome: np.ndarray, *, bins: int = 10, strategy: str = "uniform"
) -> float:
    """Maximum Calibration Error: the worst per-bin gap ``max_b |obs_b - pred_b|``.

    Unlike :func:`expected_calibration_error` this is not count-weighted, so it surfaces a small but
    badly-miscalibrated region that the average would hide -- and for exactly that reason its null
    value GROWS with ``bins``: a max over more (and therefore sparser) bins of pure binomial noise
    rises from ~0.05 to ~0.6 as ``bins`` goes 10 -> 500 under perfect calibration, with the argmax
    typically landing on the sparsest bin. A large MCE from a thin bin is what noise looks like at
    that binning; compare to :func:`calibration_null_expectation` before reading it as a defect,
    and prefer more data per bin (fewer bins, or quantile binning) when the max matters.
    """
    p, y = _probability_outcomes(prob, outcome)
    edges = _bin_edges(p, bins, strategy)
    nb = len(edges) - 1
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, nb - 1)
    worst = 0.0
    for b in range(nb):
        mask = idx == b
        if mask.any():
            worst = max(worst, abs(float(y[mask].mean()) - float(p[mask].mean())))
    return worst


def calibration_null_expectation(
    prob: np.ndarray,
    *,
    bins: int = 10,
    strategy: str = "uniform",
    norm: str = "l1",
    n_sim: int = 500,
    seed: int | RandomState | None = 0,
) -> dict[str, float]:
    """What ECE and MCE look like for a PERFECTLY calibrated forecaster at this ``prob`` profile.

    Simulates outcomes ``y*_i ~ Bernoulli(prob_i)`` -- the definition of perfect calibration at the
    observed forecasts -- and recomputes the binned statistics each time, with the same bins the
    real statistics would use. The plug-in ECE/MCE are strictly positive under this null (absolute
    values of binomial noise), so THESE numbers, not zero, are the calibrated baseline: an observed
    ECE near ``ece`` here is consistent with perfect calibration, and evidence of miscalibration
    starts around the ``*_q95`` quantiles, which give a one-sided 5% test by construction.

    Args:
        prob: ``(n,)`` the forecaster's predicted probabilities (outcomes are not needed -- the
            null generates them).
        bins, strategy, norm: the same binning controls you pass to the real statistics.
        n_sim: null simulations; at least 20. The ``*_q95`` fields are the CONSERVATIVE order
            statistic ``X_(ceil(0.95 (n_sim+1)))``, calibrated for the STRICT comparison
            ``observed > q95``: for a continuous statistic its fresh-null strict exceedance is
            exactly ``1 - k/(n_sim + 1) <= 5%`` at every accepted ``n_sim`` (4.76% at the floor
            of 20). With DISCRETE ties the two operators split -- a single forecast at p = 0.5
            makes every null ECE equal the threshold, so strict exceedance is 0 while ``>=``
            reads 1 (STAT-RR22-10) -- so compare with ``>`` and read the bound as ``<=`` (ties
            only make it more conservative). The interpolated quantile realized 9.83% at the
            floor, and ``n_sim=1`` once labeled a single draw a "q95" with 49.8% exceedance
            (STAT-RR19-10/STAT-RR21-09). The default 500 keeps the threshold's own MC noise
            small.
        seed: RNG seed.

    Returns:
        ``{'ece', 'ece_q95', 'mce', 'mce_q95', 'n_sim'}`` -- null means and 95th percentiles of the
        plug-in ECE (under ``norm``) and MCE at this forecast profile and binning.
    """
    p = _vector("prob", prob)
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("prob must contain probabilities in [0, 1]")
    if norm not in ("l1", "l2"):
        raise ValueError("norm must be 'l1' or 'l2'.")
    n_sim = _positive_int("n_sim", n_sim)
    if n_sim < 20:
        # a 95th percentile needs at least ~20 draws to exist at all: with n_sim=1 the single
        # simulation was labeled "q95" and fresh nulls exceeded it 49.8% of the time
        raise ValueError(
            f"n_sim must be at least 20 to report a 95th percentile (got {n_sim}); "
            "the q95 fields are empirical quantiles with resolution ~1/n_sim"
        )
    rng = _as_rng(seed)
    edges = _bin_edges(p, bins, strategy)
    nb = len(edges) - 1
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, nb - 1)
    n = p.shape[0]
    draws = rng.random_sample((n_sim, n)) < p  # (n_sim, n) perfectly calibrated outcomes
    ece = np.zeros(n_sim)
    mce = np.zeros(n_sim)
    for b in range(nb):
        mask = idx == b
        c = int(mask.sum())
        if c == 0:
            continue
        gap = np.abs(draws[:, mask].mean(axis=1) - float(p[mask].mean()))
        ece += (c / n) * (gap if norm == "l1" else gap * gap)
        mce = np.maximum(mce, gap)
    if norm == "l2":
        ece = np.sqrt(ece)
    # STAT-RR21-09: the interpolated quantile at the accepted n_sim floor realized 9.83%
    # exceedance while the docstring called it a one-sided 5% test. The conservative order
    # statistic X_(k) with k = ceil(0.95 (n_sim + 1)) has STRICT fresh-null exceedance exactly
    # 1 - k/(n_sim + 1) <= 5% for CONTINUOUS statistics (4.76% at the n_sim = 20 floor); with
    # discrete ties the strict exceedance only drops further (conservative), while a >= reading
    # can hit 1 -- compare with > (STAT-RR22-10/RR23-10).
    k95 = int(np.ceil(0.95 * (n_sim + 1)))
    ece_sorted = np.sort(ece)
    mce_sorted = np.sort(mce)
    return {
        "ece": float(ece.mean()),
        "ece_q95": float(ece_sorted[k95 - 1]),
        "mce": float(mce.mean()),
        "mce_q95": float(mce_sorted[k95 - 1]),
        "n_sim": float(n_sim),
    }


def top_label_confidence(prob: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a multiclass classifier to the (confidence, correct) top-label calibration problem.

    This checks calibration OF THE ARGMAX CLASS ONLY -- a deliberately narrow reduction, not THE
    multiclass calibration check. Probabilities the model puts on classes it never predicts are
    invisible to it: a forecaster that always assigns 0.4 to a class that never occurs (while its
    argmax confidences are accurate) passes top-label calibration with ECE 0. Classwise or full-
    vector calibration is a strictly stronger property and is not implemented here; when the
    non-argmax probabilities carry decisions (thresholded alerts on a minority class, expected-cost
    rankings), check those probabilities against their own outcomes directly.

    Args:
        prob: ``(n, K)`` class-probability matrix.
        labels: ``(n,)`` integer true labels.

    Returns:
        ``(confidence, correct)``: the max predicted probability per row and a 0/1 indicator of
        whether that argmax class was the true label. Feed these to :func:`reliability_curve` /
        :func:`expected_calibration_error`.
    """
    p = np.asarray(prob, dtype=float)
    labels_array = np.asarray(labels)
    if p.ndim != 2 or p.shape[0] == 0 or p.shape[1] < 2:
        raise ValueError("prob must have shape (n, K) with n >= 1 and K >= 2")
    if not np.all(np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("prob must contain finite probabilities in [0, 1]")
    if not np.allclose(p.sum(axis=1), 1.0, rtol=1e-7, atol=1e-9):
        raise ValueError("each prob row must sum to 1")
    if labels_array.ndim != 1 or labels_array.shape[0] != p.shape[0]:
        raise ValueError("labels must be one-dimensional and match the probability rows")
    try:
        labels_float = labels_array.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("labels must contain finite integer class indices") from exc
    if not np.all(np.isfinite(labels_float)) or np.any(labels_float != np.round(labels_float)):
        raise ValueError("labels must contain finite integer class indices")
    y = labels_float.astype(int)
    if np.any((y < 0) | (y >= p.shape[1])):
        raise ValueError(f"labels must lie in [0, {p.shape[1]})")
    pred = np.argmax(p, axis=1)
    confidence = p[np.arange(p.shape[0]), pred]
    correct = (pred == y).astype(float)
    return confidence, correct


def pit_values(y: np.ndarray, cdf: np.ndarray | Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Probability Integral Transform values ``u_i = F_i(y_i)``.

    Under a calibrated continuous predictive distribution the PIT values are Uniform(0, 1). Pass either
    the precomputed CDF values ``F_i(y_i)`` or a callable ``cdf(y) -> F(y)``.

    Args:
        y: ``(n,)`` realised values.
        cdf: ``(n,)`` precomputed predictive-CDF values at ``y``, or a callable applied to ``y``.

    Returns:
        ``(n,)`` PIT values in ``[0, 1]``. Invalid CDF values raise rather than being clipped.
    """
    y = _vector("y", y)
    try:
        u = np.asarray(cdf(y) if callable(cdf) else cdf, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("cdf must return finite values matching y") from exc
    if u.shape != y.shape:
        raise ValueError("cdf values must have the same one-dimensional shape as y")
    return _pit_vector(u)


def pit_ensemble(
    y: np.ndarray, forecasts: np.ndarray, *, randomize: bool = True, seed: int | RandomState | None = 0
) -> np.ndarray:
    """Rank-based PIT from a finite predictive ensemble.

    ``u_i`` is the fraction of ensemble members ``<= y_i``. With ``randomize=True`` ties are broken by
    a uniform jitter within the rank gap, which makes the PIT exactly Uniform(0, 1) under calibration
    even for discrete ensembles (the randomized PIT of Czado et al. 2009).

    Args:
        y: ``(n,)`` realised values.
        forecasts: ``(n, m)`` ensemble (``m`` draws per observation).
        randomize: jitter ties for an exactly-uniform PIT.
        seed: RNG seed when ``randomize`` is True.

    Returns:
        ``(n,)`` PIT values in ``[0, 1]``.
    """
    y = _vector("y", y)
    f = np.asarray(forecasts, dtype=float)
    if f.ndim != 2 or f.shape[0] != len(y):
        raise ValueError("forecasts must have shape (len(y), n_members)")
    if not np.all(np.isfinite(f)):
        raise ValueError("forecasts must contain only finite values")
    m = f.shape[1]
    if m == 0:
        # unguarded, this returned fabricated-looking Uniform(0,1) noise when randomize=True (u
        # reduces to just the raw jitter v) and a silent 0/0 NaN when randomize=False -- neither
        # signals "no ensemble members were provided" the way this ValueError does.
        raise ValueError("pit_ensemble requires at least one ensemble member per observation, got 0.")
    below = np.sum(f < y[:, None], axis=1)
    equal = np.sum(f == y[:, None], axis=1)
    if randomize:
        rng = _as_rng(seed)
        v = rng.rand(y.shape[0])
        u = (below + v * (equal + 1)) / (m + 1)
    else:
        u = (below + 0.5 * equal) / m
    return u


def pit_histogram(pit: np.ndarray, *, bins: int = 10) -> dict[str, np.ndarray]:
    """Histogram of PIT values with the uniform reference level.

    Args:
        pit: ``(n,)`` PIT values.
        bins: number of equal-width bins on ``[0, 1]``.

    Returns:
        ``{'counts', 'density', 'edges', 'uniform'}`` where ``density`` integrates to 1 and
        ``uniform`` is the flat reference density (``1.0``) a calibrated forecast would match.
    """
    u = _pit_vector(pit)
    bins = _positive_int("bins", bins)
    counts, edges = np.histogram(u, bins=bins, range=(0.0, 1.0))
    density = counts / (counts.sum() * (edges[1] - edges[0]))
    return {"counts": counts, "density": density, "edges": edges, "uniform": np.ones(bins)}


def pit_calibration_error(pit: np.ndarray, *, bins: int = 10) -> float:
    """Calibration error of a PIT histogram: TOTAL absolute deviation from uniform mass.

    ``sum_b |count_b/n - 1/bins|`` -- the SUM over bins (twice the total-variation distance to
    uniform, range ``[0, 2(1 - 1/bins)]``), not a per-bin mean: a threshold sized for a mean
    would be off by a factor of ``bins``. 0 when the PIT histogram is perfectly flat
    (calibrated), larger when it is U-shaped (under-dispersed) or humped (over-dispersed).
    """
    u = _pit_vector(pit)
    bins = _positive_int("bins", bins)
    counts, _ = np.histogram(u, bins=bins, range=(0.0, 1.0))
    freq = counts / counts.sum()
    return float(np.sum(np.abs(freq - 1.0 / bins)))


def interval_coverage(lower: np.ndarray, upper: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Empirical coverage and mean width of a set of prediction intervals.

    Args:
        lower: ``(n,)`` lower endpoints.
        upper: ``(n,)`` upper endpoints.
        y: ``(n,)`` realised values.

    Returns:
        ``{'coverage', 'mean_width'}`` -- the fraction of ``y`` inside ``[lower, upper]`` and the mean
        interval width. Compare ``coverage`` to the nominal level the interval was built for.
    """
    lo = _vector("lower", lower)
    hi = _vector("upper", upper)
    y = _vector("y", y)
    if lo.shape != hi.shape or lo.shape != y.shape:
        raise ValueError("lower, upper, and y must have matching shapes")
    if np.any(lo > hi):
        raise ValueError("each interval must satisfy lower <= upper")
    covered = (y >= lo) & (y <= hi)
    return {"coverage": float(covered.mean()), "mean_width": float((hi - lo).mean())}


def coverage_curve(forecasts: np.ndarray, y: np.ndarray, *, levels: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Empirical-vs-nominal coverage of central intervals across a grid of nominal levels.

    For each nominal central level ``c`` the per-observation central interval
    ``[quantile((1-c)/2), quantile((1+c)/2)]`` is read off the predictive ensemble and its empirical
    coverage is measured.

    Judge ``empirical`` against ``null_expectation``, NOT the diagonal. Plug-in quantiles of a
    finite ensemble cover BELOW nominal even when the ensemble is perfect: with m = 5 members, a
    perfectly calibrated forecast's 95% interval covers only ~0.64 -- the interval simply cannot
    reach past the sample extremes. ``null_expectation`` is the expected coverage of a perfectly
    calibrated m-member ensemble at each level, computed for a uniform reference with the same
    quantile rule. It is almost -- not exactly -- distribution-free: linear quantile interpolation
    leaves a small dependence on the ensemble's shape at very small m (measured 0.632 / 0.644 /
    0.636 for uniform / Gaussian / exponential at m = 5 and nominal 0.95, converging by m ~ 50),
    which is an order of magnitude smaller than the diagonal gap it corrects. ``empirical`` below
    the DIAGONAL at small m is expected; ``empirical`` meaningfully below ``null_expectation``
    indicates over-confidence.

    Args:
        forecasts: ``(n, m)`` predictive ensemble.
        y: ``(n,)`` realised values.
        levels: nominal central coverage levels in ``(0, 1)``; defaults to ``0.05 .. 0.95`` by 0.05.

    Returns:
        ``{'nominal', 'empirical', 'null_expectation'}`` arrays of equal length --
        ``null_expectation`` is the finite-m perfect-calibration benchmark per level.
    """
    f = np.asarray(forecasts, dtype=float)
    y = _vector("y", y)
    if f.ndim != 2 or f.shape[0] != len(y) or f.shape[1] == 0:
        raise ValueError("forecasts must have non-empty shape (len(y), n_members)")
    if not np.all(np.isfinite(f)):
        raise ValueError("forecasts must contain only finite values")
    if levels is None:
        levels = np.arange(0.05, 1.0, 0.05)
    levels = _vector("levels", levels)
    if np.any((levels <= 0.0) | (levels >= 1.0)):
        raise ValueError("levels must lie strictly between 0 and 1")
    from mixle.inference.calibration_gate import _reference_coverage_null_expectation

    emp = np.empty_like(levels)
    null_expectation = np.empty_like(levels)
    m = int(f.shape[1])
    for i, c in enumerate(levels):
        lo = np.quantile(f, (1.0 - c) / 2.0, axis=1)
        hi = np.quantile(f, (1.0 + c) / 2.0, axis=1)
        emp[i] = float(((y >= lo) & (y <= hi)).mean())
        # the same uniform-reference finite-m benchmark the calibration gate compares against
        # (GATE-3): a perfect m-member ensemble's plug-in interval covers below nominal, so the
        # honest reference at each level is this expectation, not the diagonal
        null_expectation[i] = _reference_coverage_null_expectation(m, float(c))
    return {"nominal": levels, "empirical": emp, "null_expectation": null_expectation}


def _pava(y: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Weighted non-decreasing least-squares fit, one fitted value per input location."""
    y = _vector("PAVA outcomes", y)
    w = np.ones_like(y) if weights is None else _vector("PAVA weights", weights)
    if w.shape != y.shape or np.any(w <= 0.0):
        raise ValueError("PAVA weights must match outcomes and be strictly positive")
    blocks: list[list[float]] = []  # [weighted sum, weight, start, end]
    for index, (yi, wi) in enumerate(zip(y, w)):
        cur = [float(yi * wi), float(wi), float(index), float(index + 1)]
        while blocks and blocks[-1][0] / blocks[-1][1] > cur[0] / cur[1]:
            previous = blocks.pop()
            cur = [
                previous[0] + cur[0],
                previous[1] + cur[1],
                previous[2],
                cur[3],
            ]
        blocks.append(cur)
    out = np.empty(len(y), dtype=float)
    for total, weight, start, end in blocks:
        out[int(start) : int(end)] = total / weight
    return out


class ProbabilityCalibrator:
    """Map raw scores to *calibrated probabilities* -- fit against binary outcomes.

    A raw score (a model's confidence, a self-consistency fraction, a token likelihood) need not be a
    probability of anything: it can be monotone-but-miscalibrated, or have no relationship to the
    outcome at all. This learns the transform ``score -> P(outcome = 1 | score)`` from labeled data,
    so the output *is* a probability of the event you calibrated against.

    * ``method="isotonic"`` -- monotone, non-parametric (pool-adjacent-violators). Assumes higher
      score => not-lower probability; flexible, needs enough calibration points.
    * ``method="platt"`` -- logistic ``sigmoid(a * score + b)``. Two parameters, usable on little
      data, but assumes a sigmoidal relationship. Fit against Platt's smoothed targets
      ``(N+ + 1)/(N+ + 2)`` and ``1/(N- + 2)`` rather than raw 0/1: with raw targets, perfectly
      separable calibration data drives ``a`` to infinity and the "calibrated" outputs to exact
      0/1 -- maximal overconfidence from the method that was supposed to remove it. The smoothing
      keeps the map finite and the outputs strictly inside ``(0, 1)``.

    The guarantee is conditional on HELD-OUT evaluation. The isotonic fit interpolates its own
    calibration outcomes, so re-checking calibration on the SAME data it was fit on passes by
    construction (a reliability curve of the fitted values against the fitting outcomes sits on the
    diagonal no matter how little signal the score carried); any honest check of a fitted
    calibrator -- isotonic especially -- uses data it never saw. Both methods assume the
    score/outcome relationship is stable between fitting and serving.

    A near-flat fitted curve is itself the finding: it means the raw score carried little information
    about the outcome (its "likelihood" was unrelated to the event).
    """

    def __init__(self, method: str = "isotonic") -> None:
        if method not in ("isotonic", "platt"):
            raise ValueError("method must be 'isotonic' or 'platt'")
        self.method = method
        self._fitted = False

    def fit(self, scores: Any, outcomes: Any) -> ProbabilityCalibrator:
        """Fit the score->probability map on ``scores`` with binary ``outcomes`` (0/1)."""
        s = _vector("scores", scores)
        y = _vector("outcomes", outcomes)
        if s.shape != y.shape:
            raise ValueError("scores and outcomes must have the same length")
        if s.size < 2:
            raise ValueError("need at least two calibration points")
        if np.any((y != 0.0) & (y != 1.0)):
            raise ValueError("outcomes must contain binary 0/1 values")
        if self.method == "isotonic":
            order = np.argsort(s, kind="mergesort")
            xs = s[order]
            ys = y[order]
            # Pool tied scores to their mean outcome BEFORE running PAVA, not after: running PAVA on
            # the raw (possibly tied-x) sequence and then keeping only each tied group's FIRST
            # occurrence's fitted value does not recover the group's pooled mean -- PAVA only
            # guarantees pooling actual monotonicity *violations*, so a tied-x group with a mixed,
            # non-monotone y sub-sequence (e.g. scores [1,1,1,5], outcomes [0,1,0,1]) can leave
            # different members of the SAME tied group in different PAVA blocks, and which one
            # happens to be "first" is an accident of stable-sort tie order, not the group's true
            # rate. Aggregating first (this is the standard isotonic-regression treatment of ties,
            # e.g. sklearn's IsotonicRegression) makes each unique score contribute its own mean
            # exactly once, so PAVA pools on the real per-score rate.
            uniq, inverse, counts = np.unique(xs, return_inverse=True, return_counts=True)
            group_sum = np.zeros(uniq.shape[0])
            np.add.at(group_sum, inverse, ys)
            group_mean = group_sum / counts
            fit = np.clip(_pava(group_mean, counts.astype(float)), 0.0, 1.0)
            self._x = uniq
            self._y = np.maximum.accumulate(fit)
        else:  # platt
            from scipy.optimize import minimize

            sm, ss = s.mean(), s.std() + 1e-12
            z = (s - sm) / ss  # standardize for a well-scaled logistic fit
            # Platt's smoothed regression targets (CAL-5): raw 0/1 targets make the MLE diverge on
            # separable data (a -> inf, outputs snap to exact 0/1). Regressing on the smoothed
            # values -- the posterior mean of each class's rate under a uniform prior -- bounds the
            # optimum and keeps predictions strictly inside (0, 1).
            n_pos = float(y.sum())
            n_neg = float(y.size - n_pos)
            targets = np.where(y == 1.0, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))

            def nll(theta: np.ndarray) -> float:
                a, b = theta
                logits = a * z + b
                # stable cross-entropy against the smoothed targets
                return float(np.mean(np.logaddexp(0.0, logits) - targets * logits))

            res = minimize(nll, np.array([1.0, 0.0]), method="BFGS")
            if not res.success or not np.all(np.isfinite(res.x)):
                raise RuntimeError(f"Platt calibration failed to converge: {res.message}")
            self._a, self._b, self._sm, self._ss = res.x[0], res.x[1], sm, ss
        self._fitted = True
        return self

    def predict(self, scores: Any) -> np.ndarray:
        """Calibrated probabilities for ``scores`` (clamped to ``[0, 1]``)."""
        if not self._fitted:
            raise RuntimeError("call fit(...) before predict(...)")
        s = _vector("scores", scores)
        if self.method == "isotonic":
            return np.clip(np.interp(s, self._x, self._y), 0.0, 1.0)
        z = (s - self._sm) / self._ss
        return 1.0 / (1.0 + np.exp(-(self._a * z + self._b)))

    def __call__(self, scores: Any) -> np.ndarray:
        return self.predict(scores)


def calibrate_probabilities(scores: Any, outcomes: Any, *, method: str = "isotonic") -> ProbabilityCalibrator:
    """Fit a :class:`ProbabilityCalibrator` mapping ``scores`` to ``P(outcome=1 | score)``."""
    return ProbabilityCalibrator(method).fit(scores, outcomes)


__all__ = [
    "reliability_curve",
    "expected_calibration_error",
    "maximum_calibration_error",
    "calibration_null_expectation",
    "top_label_confidence",
    "pit_values",
    "pit_ensemble",
    "pit_histogram",
    "pit_calibration_error",
    "interval_coverage",
    "coverage_curve",
    "ProbabilityCalibrator",
    "calibrate_probabilities",
]
