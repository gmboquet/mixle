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

Several functions take ``ci=True`` to attach a nonparametric bootstrap confidence band, so a
reliability diagram or ECE comes with uncertainty bands rather than a bare point estimate.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.random import RandomState


def _as_rng(seed: int | RandomState | None) -> RandomState:
    """Return a ``RandomState`` from an int seed, an existing ``RandomState``, or ``None``."""
    if isinstance(seed, RandomState):
        return seed
    return RandomState(seed)


def _bin_edges(prob: np.ndarray, bins: int, strategy: str) -> np.ndarray:
    """Equal-width (``"uniform"``) or equal-count (``"quantile"``) bin edges on ``[0, 1]``."""
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
        ci: if True attach a percentile bootstrap band on the observed frequency in each bin.
        n_boot: bootstrap resamples when ``ci`` is True.
        ci_level: central probability of the bootstrap band (e.g. 0.95).
        seed: RNG seed for the bootstrap.

    Returns:
        ``{'mean_pred', 'obs_freq', 'count', 'bin_edges'}`` (one entry per non-empty bin), plus
        ``'obs_lo'`` / ``'obs_hi'`` when ``ci`` is True.
    """
    p = np.asarray(prob, dtype=float)
    y = np.asarray(outcome, dtype=float)
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
    rooted). Zero is perfect calibration. For multiclass classifiers reduce with
    :func:`top_label_confidence` first.

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

    p = np.asarray(prob, dtype=float)
    y = np.asarray(outcome, dtype=float)
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
    badly-miscalibrated region that the average would hide.
    """
    p = np.asarray(prob, dtype=float)
    y = np.asarray(outcome, dtype=float)
    edges = _bin_edges(p, bins, strategy)
    nb = len(edges) - 1
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, nb - 1)
    worst = 0.0
    for b in range(nb):
        mask = idx == b
        if mask.any():
            worst = max(worst, abs(float(y[mask].mean()) - float(p[mask].mean())))
    return worst


def top_label_confidence(prob: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a multiclass classifier to the (confidence, correct) top-label calibration problem.

    Args:
        prob: ``(n, K)`` class-probability matrix.
        labels: ``(n,)`` integer true labels.

    Returns:
        ``(confidence, correct)``: the max predicted probability per row and a 0/1 indicator of
        whether that argmax class was the true label. Feed these to :func:`reliability_curve` /
        :func:`expected_calibration_error`.
    """
    p = np.asarray(prob, dtype=float)
    y = np.asarray(labels).astype(int)
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
        ``(n,)`` PIT values, clipped to ``[0, 1]``.
    """
    y = np.asarray(y, dtype=float)
    u = cdf(y) if callable(cdf) else np.asarray(cdf, dtype=float)
    return np.clip(u, 0.0, 1.0)


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
    y = np.asarray(y, dtype=float)
    f = np.asarray(forecasts, dtype=float)
    m = f.shape[1]
    below = np.sum(f < y[:, None], axis=1)
    equal = np.sum(f == y[:, None], axis=1)
    if randomize:
        rng = _as_rng(seed)
        v = rng.rand(y.shape[0])
        u = (below + v * (equal + 1)) / (m + 1)
    else:
        u = (below + 0.5 * equal) / m
    return np.clip(u, 0.0, 1.0)


def pit_histogram(pit: np.ndarray, *, bins: int = 10) -> dict[str, np.ndarray]:
    """Histogram of PIT values with the uniform reference level.

    Args:
        pit: ``(n,)`` PIT values.
        bins: number of equal-width bins on ``[0, 1]``.

    Returns:
        ``{'counts', 'density', 'edges', 'uniform'}`` where ``density`` integrates to 1 and
        ``uniform`` is the flat reference density (``1.0``) a calibrated forecast would match.
    """
    u = np.asarray(pit, dtype=float)
    counts, edges = np.histogram(u, bins=bins, range=(0.0, 1.0))
    density = counts / (counts.sum() * (edges[1] - edges[0]))
    return {"counts": counts, "density": density, "edges": edges, "uniform": np.ones(bins)}


def pit_calibration_error(pit: np.ndarray, *, bins: int = 10) -> float:
    """Calibration error of a PIT histogram: mean absolute deviation from uniform mass.

    ``sum_b |count_b/n - 1/bins|`` -- 0 when the PIT histogram is perfectly flat (calibrated), larger
    when it is U-shaped (under-dispersed) or humped (over-dispersed).
    """
    u = np.asarray(pit, dtype=float)
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
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    y = np.asarray(y, dtype=float)
    covered = (y >= lo) & (y <= hi)
    return {"coverage": float(covered.mean()), "mean_width": float((hi - lo).mean())}


def coverage_curve(forecasts: np.ndarray, y: np.ndarray, *, levels: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Empirical-vs-nominal coverage of central intervals across a grid of nominal levels.

    For each nominal central level ``c`` the per-observation central interval
    ``[quantile((1-c)/2), quantile((1+c)/2)]`` is read off the predictive ensemble and its empirical
    coverage is measured. A calibrated forecast traces the diagonal ``empirical == nominal``; bowing
    below the diagonal means the intervals are too narrow (over-confident).

    Args:
        forecasts: ``(n, m)`` predictive ensemble.
        y: ``(n,)`` realised values.
        levels: nominal central coverage levels in ``(0, 1)``; defaults to ``0.05 .. 0.95`` by 0.05.

    Returns:
        ``{'nominal', 'empirical'}`` arrays of equal length.
    """
    f = np.asarray(forecasts, dtype=float)
    y = np.asarray(y, dtype=float)
    if levels is None:
        levels = np.arange(0.05, 1.0, 0.05)
    levels = np.asarray(levels, dtype=float)
    emp = np.empty_like(levels)
    for i, c in enumerate(levels):
        lo = np.quantile(f, (1.0 - c) / 2.0, axis=1)
        hi = np.quantile(f, (1.0 + c) / 2.0, axis=1)
        emp[i] = float(((y >= lo) & (y <= hi)).mean())
    return {"nominal": levels, "empirical": emp}


def _pava(y: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators: the non-decreasing least-squares fit to ``y`` (in order)."""
    blocks: list[list[float]] = []  # each block is [sum, count]
    for yi in y:
        cur = [float(yi), 1.0]
        while blocks and blocks[-1][0] / blocks[-1][1] >= cur[0] / cur[1]:
            s, c = blocks.pop()
            cur[0] += s
            cur[1] += c
        blocks.append(cur)
    out = np.empty(len(y))
    i = 0
    for s, c in blocks:
        k = int(round(c))
        out[i : i + k] = s / c
        i += k
    return out


class ProbabilityCalibrator:
    """Map raw scores to *calibrated probabilities* -- fit against binary outcomes.

    A raw score (a model's confidence, a self-consistency fraction, a token likelihood) need not be a
    probability of anything: it can be monotone-but-miscalibrated, or have no relationship to the
    outcome at all. This learns the transform ``score -> P(outcome = 1 | score)`` from labeled data,
    so the output *is* a probability of the event you calibrated against.

    * ``method="isotonic"`` -- monotone, non-parametric (pool-adjacent-violators). Assumes higher
      score => not-lower probability; flexible, needs enough calibration points.
    * ``method="platt"`` -- logistic ``sigmoid(a * score + b)``. Two parameters, robust on little data,
      but assumes a sigmoidal relationship.

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
        s = np.asarray(scores, dtype=float).reshape(-1)
        y = np.asarray(outcomes, dtype=float).reshape(-1)
        if s.shape != y.shape:
            raise ValueError("scores and outcomes must have the same length")
        if s.size < 2:
            raise ValueError("need at least two calibration points")
        if self.method == "isotonic":
            order = np.argsort(s, kind="mergesort")
            xs = s[order]
            fit = np.clip(_pava(y[order]), 0.0, 1.0)
            # collapse ties to a strictly increasing support for interpolation
            uniq, idx = np.unique(xs, return_index=True)
            self._x = uniq
            self._y = np.maximum.accumulate(fit[idx])
        else:  # platt
            from scipy.optimize import minimize

            sm, ss = s.mean(), s.std() + 1e-12
            z = (s - sm) / ss  # standardize for a well-scaled logistic fit

            def nll(theta: np.ndarray) -> float:
                a, b = theta
                logits = a * z + b
                # stable BCE
                return float(np.mean(np.logaddexp(0.0, logits) - y * logits))

            res = minimize(nll, np.array([1.0, 0.0]), method="BFGS")
            self._a, self._b, self._sm, self._ss = res.x[0], res.x[1], sm, ss
        self._fitted = True
        return self

    def predict(self, scores: Any) -> np.ndarray:
        """Calibrated probabilities for ``scores`` (clamped to ``[0, 1]``)."""
        if not self._fitted:
            raise RuntimeError("call fit(...) before predict(...)")
        s = np.asarray(scores, dtype=float).reshape(-1)
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
